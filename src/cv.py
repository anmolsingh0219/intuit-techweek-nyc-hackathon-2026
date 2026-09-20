"""Rigorous cross-validation engine: out-of-fold (OOF) predictions for honest
evaluation, a stacked ensemble, a recovery model, and OOF P&L.

Why GroupKFold on business_id: a business can submit multiple applications, so a
random split would leak a business across train/holdout. Grouping by business
makes the OOF estimate honest — the foundation for every model/threshold choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from . import features, models, npv

N_WEEKS = 13


def oof_predict(backbone: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                cols: list[str], k: int = 5, monotone: bool = True) -> tuple[np.ndarray, list]:
    """Return (oof_pd, fitted_models). oof_pd[i] is predicted by a model that did
    NOT see business i's group."""
    oof = np.zeros(len(y))
    fitted = []
    gkf = GroupKFold(n_splits=k)
    for tr_idx, va_idx in gkf.split(X, y, groups):
        m = models.make_backbone(backbone, cols, seed=0, monotone=monotone)
        m.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = models.proba(m, X.iloc[va_idx])
        fitted.append(m)
    return oof, fitted


def stack_oof(base_oof: dict[str, np.ndarray], y: np.ndarray) -> tuple[np.ndarray, LogisticRegression]:
    """Meta-learner (logistic) on base OOF predictions -> stacked OOF prediction.
    Fit on logit(base preds); honest because bases are already OOF."""
    Z = _stack_matrix(base_oof)
    meta = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y)
    return meta.predict_proba(Z)[:, 1], meta


def _stack_matrix(base: dict[str, np.ndarray]) -> np.ndarray:
    cols = [np.clip(base[k], 1e-6, 1 - 1e-6) for k in sorted(base)]
    logit = [np.log(c / (1 - c)) for c in cols]
    return np.column_stack(logit)


def apply_stack(meta: LogisticRegression, base_preds: dict[str, np.ndarray]) -> np.ndarray:
    return meta.predict_proba(_stack_matrix(base_preds))[:, 1]


# --------------------------------------------------------------------------- #
# Recovery model + timing for E[NPV]
# --------------------------------------------------------------------------- #

def fit_recovery(labeled: pd.DataFrame, fstate) -> object:
    """Predict recovery fraction (recovered/principal) for defaulted loans."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    d = labeled[labeled["repayment_status"] == "defaulted"].copy()
    Xd, _ = features.build_features(d, fstate)
    frac = (d["final_recovered_amount"] / d["requested_amount"]).clip(0, 1).fillna(0).to_numpy()
    return HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05).fit(Xd, frac)


def recovery_rate(model, X) -> np.ndarray:
    return np.clip(model.predict(X), 0, 1)


def week_probs_global(labeled: pd.DataFrame, fstate, sample_weight=None):
    """Single conditional timing model on all defaulters (for OOF P&L eval)."""
    d = labeled[labeled["default_flag"] == 1]
    Xd, _ = features.build_features(d, fstate)
    wk = np.ceil(d["days_to_default"].to_numpy() / 7.0).clip(1, N_WEEKS).astype(int)
    classes = np.array(sorted(np.unique(wk)))
    cls = {c: i for i, c in enumerate(classes)}
    yk = np.array([cls[v] for v in wk])
    from xgboost import XGBClassifier
    tm = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, objective="multi:softprob", num_class=len(classes),
                       eval_metric="mlogloss", tree_method="hist", random_state=7, n_jobs=0)
    tm.fit(Xd, yk)
    day_per_week = np.array([7 * k - 3 for k in range(1, N_WEEKS + 1)], dtype=float)
    for k in range(1, N_WEEKS + 1):
        sel = wk == k
        if sel.any():
            day_per_week[k - 1] = float(d["days_to_default"].to_numpy()[sel].mean())
    return tm, classes, day_per_week


def expand_week_probs(tm, classes, X) -> np.ndarray:
    p = tm.predict_proba(X)
    full = np.zeros((X.shape[0], N_WEEKS))
    full[:, classes - 1] = p
    s = full.sum(axis=1, keepdims=True)
    return np.divide(full, s, out=np.zeros_like(full), where=s > 0)


def enpv_from_pd(pd_pt, week_probs, R, day_per_week, rec_rate) -> np.ndarray:
    pi = pd_pt[:, None] * week_probs
    return npv.expected_npv_weekly(R, pi, day_per_week, rec_rate)
