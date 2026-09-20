"""Deliverable C — interventional (counterfactual) PD:  P(default | do(X = v)).

The brief's distinction:  P(y | do(X=x))  !=  P(y | X=x)  when X is confounded.
A purely NAIVE approach patches one model input and re-predicts, leaving every
feature DERIVED from X stale and self-inconsistent.

Our STRUCTURAL approach: set the raw feature to v, then **rebuild the feature
vector**, which deterministically propagates the intervention through every
descendant we know exactly — engineered ratios/interactions/squares, the one-hot
encodings, and the provided requested/observed-revenue ratio. This is the part of
the causal effect we can compute without error. We do NOT claim to de-confound the
remaining observational associations (the data is observational and features are
anonymized, precluding a credible full DAG); monotonic constraints act as a weak
causal-direction prior, and C intervals are widened to reflect that residual,
un-testable causal uncertainty. (Full reasoning in writeup S3.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features, models

# Provided columns that are deterministic functions of other raw columns and must
# be recomputed under intervention (engineered features rebuild automatically).
_REQ, _OBSREV = "requested_amount", "observed_monthly_revenue_avg_3mo"
_REQ_OBS_RATIO = "requested_amount_to_observed_revenue"


def build_cf_rows(pool: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """One intervened raw row per query (in query order)."""
    idx = pool.set_index("applicant_id", drop=False)
    rows = []
    for _, q in queries.iterrows():
        r = idx.loc[q["applicant_id"]].copy()
        r[q["feature_name"]] = q["intervention_value"]
        # recompute the provided deterministic ratio if its inputs are usable
        a, b = r.get(_REQ), r.get(_OBSREV)
        if pd.notna(a) and pd.notna(b) and b != 0:
            r[_REQ_OBS_RATIO] = a / b
        rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


def _bag_pd(model, X) -> np.ndarray:
    return np.column_stack([models.proba(m, X) for m in model.pd_bag]).mean(axis=1)


def structural_cf(model, calibrator, cf_raw: pd.DataFrame) -> np.ndarray:
    """Calibrated counterfactual PD with full deterministic propagation."""
    X, _ = features.build_features(cf_raw, model.fstate)
    return calibrator.predict(_bag_pd(model, X))


def naive_cf(model, calibrator, pool: pd.DataFrame, queries: pd.DataFrame) -> np.ndarray:
    """No-propagation baseline: patch only the single model input (engineered
    descendants left stale). Used to quantify what propagation corrects."""
    pos = {aid: i for i, aid in enumerate(pool["applicant_id"].to_numpy())}
    X0, _ = features.build_features(pool, model.fstate)
    out = np.empty(len(queries))
    for j, (_, q) in enumerate(queries.iterrows()):
        x = X0.iloc[[pos[q["applicant_id"]]]].copy()
        f, v = q["feature_name"], q["intervention_value"]
        if f in x.columns:                       # direct numeric model input
            x[f] = v
        else:                                    # nominal -> flip one-hot dummies
            for c in [c for c in x.columns if c.startswith(f + "=")]:
                x[c] = 0
            col = f"{f}={int(v)}"
            if col in x.columns:
                x[col] = 1
        out[j] = calibrator.predict(_bag_pd(model, x))[0]
    return out
