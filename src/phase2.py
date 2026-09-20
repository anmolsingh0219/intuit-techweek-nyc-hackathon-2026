"""Phase 2 — calibrate PD + build conformal 90% intervals for A and B.

Uses the Phase-1-selected model (bagged logistic). Steps:
  1. Fit isotonic PD calibration on validation; report Brier/gap improvement.
  2. Re-derive the E[NPV] decision with CALIBRATED PD (better risk -> better folds).
  3. Tune one-parameter conformal intervals for A (PD) and B (trajectory) to ~90%
     coverage on validation; report held-out coverage + mean width.
  4. Write calibrated A & B; validate.

Run:  python run.py phase2   (Phase 1 establishes logistic as the winner)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import (calibrate, cohorts, data, evaluate, npv, paths, submission,
               survival, validate)

N_WEEKS = cohorts.N_WEEKS


def _cohort_cdr(cif_rows, y, defweek, approved, cohort, w, n_weeks=N_WEEKS):
    """Predicted & realized cumulative-default curves for one cohort's approved set."""
    sel = approved & (cohort == w)
    n = int(sel.sum())
    if n == 0:
        return None
    ages = np.arange(1, n_weeks + 1)
    pred = cif_rows[sel].mean(axis=0)
    real = np.array([np.mean((y[sel] == 1) & (defweek[sel] <= a) & (defweek[sel] >= 1)) for a in ages])
    return pred, real, n


def run() -> bool:
    print("=" * 72, "\nPHASE 2 — calibration & uncertainty (S_cal = 0.20)\n" + "=" * 72)
    tr, va, te = data.load_train(), data.load_validation(), data.load_test()
    rec = survival.recovery_rate_from(tr)
    labeled = tr[data.build_survival_label(tr)["has_label"].to_numpy()].copy()
    val = va[va["default_flag"].notna()].copy()
    y_val = val["default_flag"].astype(int).to_numpy()
    R_val = val["requested_amount"].to_numpy(float)

    # Phase-1 winner: bagged logistic, monotone, conditional timing for NPV.
    # NOTE (P6): we tested appending a monotone-HGB bag (logistic+HGB blend). It looked
    # better in a stripped bake-off (marginal timing, single fit) but REGRESSED in the
    # full pipeline -- it widened the A conformal band 0.078->0.100 (worse per-bin
    # calibration -> worse S_cal) without improving P&L. Kept the pure logistic.
    model = survival.RiskModel(backbone="logistic", k_bag=5, recovery_rate=rec).fit(labeled)
    raw_val, _ = model.predict_pd(val)
    model.timing_mode = "conditional"
    wk_cond_val = model.predict_week_probs(val)

    # ---- 1. PD point calibration: pick method by held-out Brier; tune intervals honestly ----
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(val)); half = len(val) // 2
    ca, ho = perm[:half], perm[half:]
    method, ho_brier = calibrate.select_pd_calibrator(raw_val[ca], y_val[ca], raw_val[ho], y_val[ho])
    cal_h = calibrate.fit_pd_calibrator(raw_val[ca], y_val[ca], method)
    cal_ho = calibrate.apply_calibrator(cal_h, raw_val[ho])
    print(f"\n[1] PD calibration (method='{method}', chosen by held-out Brier):")
    print("   raw  ", calibrate.calibration_report(raw_val[ho], y_val[ho]))
    print("   calib", calibrate.calibration_report(cal_ho, y_val[ho]))
    kA = calibrate.tune_pd_interval(cal_ho, y_val[ho])          # honest: tuned on held-out
    print(f"   A-interval k={kA['k']:.2f} floor={kA['floor']} "
          f"| held-out coverage={kA['coverage']:.2f} width={kA['width']:.3f}")

    # final calibrator on ALL of validation (best use of data for the submission)
    cal = calibrate.fit_pd_calibrator(raw_val, y_val, method)
    cal_val = calibrate.apply_calibrator(cal, raw_val)

    # ---- 2. decision with calibrated PD ----
    pi_val = cal_val[:, None] * wk_cond_val
    e_npv_val = npv.expected_npv_weekly(R_val, pi_val, model.day_per_week, rec)
    tb = evaluate.tune_buffer(val, e_npv_val)
    tau = tb["tau"]
    pnl_cal = evaluate.policy_pnl(val, (e_npv_val > tau).astype(int))
    print(f"\n[2] decision w/ calibrated PD: tau=${tau:,.0f} "
          f"approve={pnl_cal['approve_rate']:.3f} val_P&L=${pnl_cal['total_pnl']:,.0f}")

    # ---- 3. B trajectory intervals (tune on val approved set) ----
    coh_val = cohorts.assign_cohort(val).to_numpy()
    appr_val = e_npv_val > tau
    defweek_val = np.where(val["default_flag"].to_numpy() == 1,
                           np.ceil(val["days_to_default"].to_numpy() / 7.0), 0)
    defweek_val = np.clip(np.nan_to_num(defweek_val), 0, N_WEEKS).astype(int)
    model.timing_mode = "marginal"
    cif_val = np.cumsum(cal_val[:, None] * model.predict_week_probs(val), axis=1)
    preds, reals, ns = [], [], []
    for w in range(1, N_WEEKS + 1):
        out = _cohort_cdr(cif_val, y_val, defweek_val, appr_val, coh_val, w)
        if out:
            p, r, n = out
            preds.append(p); reals.append(r); ns.append(np.full(N_WEEKS, n))
    kB = calibrate.tune_b_interval(np.concatenate(preds), np.concatenate(reals), np.concatenate(ns))
    print(f"[3] B-interval k={kB['k']:.2f} | val cohort/age coverage={kB['coverage']:.2f}")

    # ---- 4. score pool, write calibrated A & B ----
    pool = pd.concat([va, te], ignore_index=True)
    coh = cohorts.assign_cohort(pool).to_numpy()
    raw_pool, _ = model.predict_pd(pool)
    cal_pool = calibrate.apply_calibrator(cal, raw_pool)

    model.timing_mode = "conditional"
    pi_pool = cal_pool[:, None] * model.predict_week_probs(pool)
    e_npv = npv.expected_npv_weekly(pool["requested_amount"].to_numpy(float),
                                    pi_pool, model.day_per_week, rec)
    decision = (e_npv > tau).astype(int)
    loA, hiA = calibrate.pd_intervals(cal_pool, kA["k"], kA["floor"])
    A = pd.DataFrame({"applicant_id": pool["applicant_id"].astype(str), "decision": decision,
                      "predicted_pd": cal_pool, "pd_lower_90": loA, "pd_upper_90": hiA})
    submission.write_a(A)

    model.timing_mode = "marginal"
    cif_pool = np.cumsum(cal_pool[:, None] * model.predict_week_probs(pool), axis=1)
    overall = cif_pool[decision == 1].mean(axis=0) if (decision == 1).any() else cif_pool.mean(axis=0)
    rows = []
    for w in range(1, N_WEEKS + 1):
        sel = (coh == w) & (decision == 1)
        n = int(sel.sum())
        point = cif_pool[sel].mean(axis=0) if n else overall
        se = np.sqrt(np.clip(point * (1 - point), 0, None) / max(n, 1))
        half = np.maximum(kB["k"] * se, kB["floor"])
        for a in range(1, N_WEEKS + 1):
            p = float(point[a - 1]); h = float(half[a - 1])
            rows.append((w, a, p, max(0.0, p - h), min(1.0, p + h)))
    submission.write_b(pd.DataFrame(rows, columns=submission.COLUMNS_B))

    print(f"\nscored {len(pool):,} | approve={decision.mean():.3f} | mean PD={cal_pool.mean():.3f}")
    print(f"A held-out coverage={kA['coverage']:.2f} | A interval mean width={(hiA - loA).mean():.3f} "
          f"| B val coverage={kB['coverage']:.2f} | B interval mean width="
          f"{np.mean([r[4] - r[3] for r in rows]):.3f}")

    paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths.CACHE_DIR / "phase2_preds.npz",
                        applicant_id=pool["applicant_id"].astype(str).to_numpy(),
                        cohort=coh, decision=decision, pd_cal=cal_pool, e_npv=e_npv,
                        tau=tau, kA=kA["k"], kB=kB["k"], recovery_rate=rec)

    print("\n[validate]")
    ok = validate.run()
    print("\n" + "=" * 72)
    print("PHASE 2:", "OK — calibrated A & B written and validated" if ok else "FAILED")
    print("=" * 72)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
