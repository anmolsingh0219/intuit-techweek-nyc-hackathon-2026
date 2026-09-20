"""Phase 1 (+improvements A/B/C) — fit the core A+B risk model and write A & B.

Improvements over the first cut:
  A. Monotonic constraints + stronger regularization (models.py) so the model
     extrapolates sanely into the label-scarce risky region.
  B. Tuned approval buffer: approve iff E[NPV] > tau, tau chosen on validation P&L.
  C. Model bake-off across families (XGB / LightGBM / CatBoost / HistGB / logistic)
     + ensemble of the top two.

Run:  python run.py phase1
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cohorts, data, evaluate, features, models, paths, submission, survival, validate

Z = 1.645
BAKEOFF_FAMILIES = ["xgb", "lgbm", "catboost", "hgb", "logistic"]


def _interval(point, sd, floor=0.02):
    half = np.maximum(Z * np.maximum(sd, 0.0), floor)
    return np.clip(point - half, 0, 1), np.clip(point + half, 0, 1)


def _val_labeled(va):
    return va[va["default_flag"].notna()].copy()


def _eval_on_val(model, val_lab):
    """Return AUC, Brier, tuned-tau P&L for a fitted RiskModel."""
    y = val_lab["default_flag"].astype(int).to_numpy()
    pd_pt, _ = model.predict_pd(val_lab)
    e_npv = model.expected_npv(val_lab)
    m = evaluate.pd_metrics(y, pd_pt)
    tb = evaluate.tune_buffer(val_lab, e_npv)
    return {"auc": m["auc"], "brier": m["brier"], "calib_gap": m["calib_gap"],
            "tau": tb["tau"], "pnl": tb["pnl_at_tau"], "pnl0": tb["pnl_at_0"],
            "approve_rate": tb["approve_rate"]}


def run() -> bool:
    print("=" * 72, "\nPHASE 1 — core A+B model  (A: monotone+robust, B: tuned tau, C: bake-off)\n" + "=" * 72)
    tr, va, te = data.load_train(), data.load_validation(), data.load_test()
    rec = survival.recovery_rate_from(tr)
    labeled = tr[data.build_survival_label(tr)["has_label"].to_numpy()].copy()
    val_lab = _val_labeled(va)
    print(f"labeled train: {len(labeled):,} | val labeled: {len(val_lab):,} | recovery={rec:.4f}")

    # ---- C: bake-off (single-bag fit per family) ----
    print("\n[C] Model bake-off (monotone, k_bag=1):")
    rows = {}
    fitted = {}
    for fam in BAKEOFF_FAMILIES:
        try:
            m = survival.RiskModel(backbone=fam, recovery_rate=rec, k_bag=1).fit(labeled)
            r = _eval_on_val(m, val_lab)
            fitted[fam] = m
            rows[fam] = r
        except Exception as e:  # keep going if a lib misbehaves
            print(f"   {fam:9s} FAILED: {e}")
    table = pd.DataFrame(rows).T[["auc", "brier", "calib_gap", "tau", "pnl0", "pnl", "approve_rate"]]
    print(table.round(4).to_string())

    ranked = table.sort_values(["pnl", "auc"], ascending=False).index.tolist()
    top2 = ranked[:2]
    print(f"\nranked by val P&L: {ranked}  ->  ensemble top-2: {top2}")

    # ---- final model: best of {top-1 bagged} vs {top-2 ensemble} on val P&L ----
    print(f"\n[final] candidates: single '{top2[0]}' (k=5)  vs  ensemble {top2} (k=3+3)")
    cand = {}
    m1 = survival.RiskModel(backbone=top2[0], recovery_rate=rec, k_bag=5).fit(labeled)
    cand["single"] = (m1, _eval_on_val(m1, val_lab))
    m2 = survival.RiskModel(backbone=top2[0], recovery_rate=rec, k_bag=3).fit(labeled)
    m2.add_pd_backbone(labeled, top2[1], k=3)
    cand["ensemble"] = (m2, _eval_on_val(m2, val_lab))
    for nm, (_, rr) in cand.items():
        print(f"   {nm:9s} AUC={rr['auc']:.4f} Brier={rr['brier']:.4f} "
              f"tau=${rr['tau']:,.0f} P&L@tau=${rr['pnl']:,.0f} approve={rr['approve_rate']:.3f}")
    # prefer the simpler single model unless the ensemble clearly beats it (>0.3%)
    ps, pe = cand["single"][1]["pnl"], cand["ensemble"][1]["pnl"]
    bestname = "ensemble" if pe > ps * 1.003 else "single"
    model, fin = cand[bestname]
    tau = fin["tau"]
    final_desc = top2[0] if bestname == "single" else "+".join(top2)
    print(f"   >>> selected '{bestname}' [{final_desc}]  P&L@tau=${fin['pnl']:,.0f}")

    # ---- B timing: conditional (x-dependent) vs marginal, by val trajectory MAE ----
    # A's NPV decision always uses CONDITIONAL timing (risky loans default earlier ->
    # costlier, which must drive approve/decline). B's cohort curve uses whichever
    # validates better on the aggregate (cohort-level) trajectory.
    print("\n[B] timing model — validating Deliverable-B accuracy:")
    b_mode, best_mae = "conditional", np.inf
    for mode in ("conditional", "marginal"):
        model.timing_mode = mode
        tm = evaluate.trajectory_metrics(model, val_lab)
        print(f"   {mode:11s} traj_MAE={tm['traj_mae']:.4f}  timing_MAE={tm['timing_mae']:.4f}  "
              f"final pred/real={tm['final_pred']:.3f}/{tm['final_real']:.3f}")
        if tm["traj_mae"] < best_mae:
            best_mae, b_mode = tm["traj_mae"], mode
    print(f"   >>> B trajectory uses '{b_mode}' (val traj_MAE={best_mae:.4f}); "
          f"A decisions use 'conditional'")

    pool = pd.concat([va, te], ignore_index=True)
    cohort = cohorts.assign_cohort(pool).to_numpy()
    Xpool, _ = features.build_features(pool, model.fstate)

    # ---- Deliverable A (conditional timing for the NPV decision) ----
    model.timing_mode = "conditional"
    pd_pt, pd_sd, pi = model.predict_pi(pool)
    e_npv = model.expected_npv(pool)
    decision = (e_npv > tau).astype(int)
    lo, hi = _interval(pd_pt, pd_sd)
    print(f"\nscored {len(pool):,} | approve={decision.mean():.3f} | mean PD={pd_pt.mean():.3f}")
    A = pd.DataFrame({"applicant_id": pool["applicant_id"].astype(str), "decision": decision,
                      "predicted_pd": pd_pt, "pd_lower_90": lo, "pd_upper_90": hi})
    submission.write_a(A)

    # ---- Deliverable B (best-MAE timing for the cohort trajectory) ----
    model.timing_mode = b_mode
    cif = model.predict_cif(pool)
    cumwk = np.cumsum(model.predict_week_probs(pool), axis=1)
    bag_cif = [models.proba(m, Xpool)[:, None] * cumwk for m in model.pd_bag]
    overall_cif = cif[decision == 1].mean(axis=0) if (decision == 1).any() else cif.mean(axis=0)

    rows_b = []
    for w in range(1, cohorts.N_WEEKS + 1):
        sel = (cohort == w) & (decision == 1)
        n = int(sel.sum())
        if n == 0:
            point, se = overall_cif, np.full(cohorts.N_WEEKS, 0.05)
        else:
            point = cif[sel].mean(axis=0)
            model_sd = np.std([bc[sel].mean(axis=0) for bc in bag_cif], axis=0)
            binom_se = np.sqrt(np.clip(point * (1 - point), 0, None) / n)
            se = np.sqrt(model_sd ** 2 + binom_se ** 2)
        for a in range(1, cohorts.N_WEEKS + 1):
            p = float(point[a - 1]); h = max(Z * float(se[a - 1]), 0.01)
            rows_b.append((w, a, p, max(0.0, p - h), min(1.0, p + h)))
    submission.write_b(pd.DataFrame(rows_b, columns=submission.COLUMNS_B))
    print("approved-cohort sizes:",
          {int(w): int(((cohort == w) & (decision == 1)).sum()) for w in range(1, 14)})

    # ---- persist for Phase 2/3 ----
    paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths.CACHE_DIR / "phase1_preds.npz",
        applicant_id=pool["applicant_id"].astype(str).to_numpy(),
        cohort=cohort, decision=decision, pd_point=pd_pt, pd_std=pd_sd,
        pi=pi, cif=cif, e_npv=e_npv, day_per_week=model.day_per_week,
        tau=tau, final=final_desc, selected=bestname, b_timing=b_mode, recovery_rate=rec,
    )

    print("\n[validate]")
    ok = validate.run()
    print("\n" + "=" * 72)
    print("PHASE 1:", "OK — A & B written and validated" if ok else "FAILED")
    print("=" * 72)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
