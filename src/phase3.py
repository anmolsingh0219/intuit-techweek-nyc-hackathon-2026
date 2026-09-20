"""Phase 3 — Deliverable C: interventional counterfactual PDs for the ~900 queries.

Structural counterfactual = set raw feature -> rebuild vector (deterministic
propagation) -> calibrated bagged-logistic PD. Intervals reuse A's conformal
(k, floor) widened to reflect un-testable causal uncertainty. Also reports the
naive-vs-structural gap and the biggest causal movers (writeup S3 material).

Run:  python run.py phase3
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import (calibrate, causal, data, features, paths, pipeline, submission,
               survival, validate)

CAUSAL_EXTRA_FLOOR = 0.03   # extra interval width for un-falsifiable causal assumptions


def run() -> bool:
    print("=" * 72, "\nPHASE 3 — Deliverable C: counterfactual PD  P(default | do(X=v))\n" + "=" * 72)
    tr, va, te = data.load_train(), data.load_validation(), data.load_test()
    rec = survival.recovery_rate_from(tr)
    labeled = tr[data.build_survival_label(tr)["has_label"].to_numpy()].copy()
    val = va[va["default_flag"].notna()].copy()
    queries = pd.read_csv(paths.INTERVENTION_QUERIES)
    pool = pd.concat([va, te], ignore_index=True)
    print(f"queries: {len(queries)} on {queries['applicant_id'].nunique()} applicants "
          f"across {queries['feature_name'].nunique()} features")

    # C uses a BOUNDED tree (HistGB), not the linear A/B model: linear extrapolation
    # makes single-feature counterfactuals implausibly large (PD swings to ~0.8);
    # trees stay locally bounded -> credible interventional magnitudes, and handle
    # the no-bank-feed applicants (NaN) natively.
    model, cal, kA, tau, method = pipeline.fit_calibrated_model(labeled, val, rec, backbone="hgb")
    print(f"model: bagged monotone HistGB | calibrator={method} | A-interval k={kA['k']:.2f} floor={kA['floor']}")

    # baseline (no intervention) calibrated PD per applicant, mapped to each query
    Xpool, _ = features.build_features(pool, model.fstate)
    base_pool = cal.predict(causal._bag_pd(model, Xpool))
    base_by_id = dict(zip(pool["applicant_id"], base_pool))
    baseline = queries["applicant_id"].map(base_by_id).to_numpy()

    # counterfactuals
    cf_raw = causal.build_cf_rows(pool, queries)
    cf = causal.structural_cf(model, cal, cf_raw)
    naive = causal.naive_cf(model, cal, pool, queries)

    # intervals: A's conformal, widened for causal uncertainty
    floor_c = kA["floor"] + CAUSAL_EXTRA_FLOOR
    lo, hi = calibrate.pd_intervals(cf, kA["k"], floor_c)

    C = pd.DataFrame({"query_id": queries["query_id"], "predicted_pd_cf": cf,
                      "pd_cf_lower_90": lo, "pd_cf_upper_90": hi})
    submission.write_c(C)

    # ---- diagnostics for the writeup ----
    eff = cf - baseline
    print(f"\ncounterfactual PD: mean={cf.mean():.3f}  | mean |effect vs baseline|={np.abs(eff).mean():.3f}"
          f"  | mean |structural - naive|={np.abs(cf - naive).mean():.4f}")
    print(f"interval mean width={ (hi - lo).mean():.3f} (floor={floor_c:.2f})")
    qd = queries.assign(effect=eff, prop=np.abs(cf - naive))
    top = (qd.groupby("feature_name")
             .agg(n=("effect", "size"), mean_effect=("effect", "mean"),
                  mean_abs_effect=("effect", lambda s: s.abs().mean()),
                  propagation=("prop", "mean"))
             .sort_values("mean_abs_effect", ascending=False))
    print("\nbiggest causal movers (mean signed effect on PD, by intervened feature):")
    print(top.head(10).round(4).to_string())

    paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths.CACHE_DIR / "phase3_preds.npz",
                        query_id=queries["query_id"].to_numpy(), cf=cf, baseline=baseline,
                        naive=naive, lo=lo, hi=hi)

    print("\n[validate]")
    ok = validate.run()
    print("\n" + "=" * 72)
    print("PHASE 3:", "OK — C written and validated" if ok else "FAILED")
    print("=" * 72)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
