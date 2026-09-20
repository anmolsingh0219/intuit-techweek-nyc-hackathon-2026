"""PD model backbones + monotonic constraints (Phase-1 improvements A & C).

Why monotonic constraints (Option A): labels exist only for the prior-approved
(safe) region, so the model must EXTRAPOLATE into the risky, label-scarce region.
Forcing economically-unambiguous directions (more debt/utilization/inquiries ->
never *less* predicted risk; more revenue/cash -> never *more*) makes that
extrapolation behave sanely and is exactly what a regulator expects (writeup S3).

Backbones (Option C bake-off): gradient boosters (XGB / LightGBM / CatBoost /
sklearn HistGB) plus a logistic-regression baseline, all behind one interface:
`.fit(X, y, sample_weight) -> self` and `.predict_proba(X)[:, 1]`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Economically-unambiguous monotone directions w.r.t. PD (probability of default).
# +1: feature up  => risk never decreases.   -1: feature up => risk never increases.
MONOTONE: dict[str, int] = {
    # riskier when higher
    "aggregate_credit_utilization": +1,
    "recent_inquiries_count_6mo": +1,
    "existing_debt_obligations": +1,
    "invoice_payment_delinquency_rate": +1,
    "multi_lender_inquiry_count_30d": +1,
    "observed_overdraft_count_3mo": +1,
    "prior_loans_default_count": +1,
    "prior_default_rate": +1,          # engineered
    "req_to_stated_rev": +1,           # engineered leverage
    "debt_to_stated_rev": +1,          # engineered
    "loan_to_observed_annual_rev": +1,
    "draw_to_daily_rev": +1,           # ACH affordability
    "debt_service_ratio": +1,
    "inquiry_intensity": +1,
    "overdraft_per_month": +1,
    # safer when higher
    "prior_underwriter_score": -1,
    "observed_monthly_revenue_avg_3mo": -1,
    "observed_cash_balance_p10": -1,
    "payroll_regularity_score": -1,
    "vintage_years": -1,
    "stated_time_in_business": -1,
    "cash_runway_days": -1,
}


def monotone_vector(columns: list[str]) -> list[int]:
    """Per-column constraint list aligned to the feature matrix column order."""
    return [MONOTONE.get(c, 0) for c in columns]


# --------------------------------------------------------------------------- #
# Backbone factories. `columns` is the fitted feature order (for monotone maps).
# --------------------------------------------------------------------------- #

def make_xgb(columns, seed=0, monotone=True):
    from xgboost import XGBClassifier
    mc = {c: MONOTONE[c] for c in columns if c in MONOTONE} if monotone else None
    return XGBClassifier(
        n_estimators=450, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=6,
        reg_lambda=2.0, reg_alpha=0.5, gamma=0.5,
        tree_method="hist", eval_metric="logloss",
        monotone_constraints=mc, random_state=seed, n_jobs=0,
    )


def make_lgbm(columns, seed=0, monotone=True):
    from lightgbm import LGBMClassifier
    mc = monotone_vector(columns) if monotone else None
    return LGBMClassifier(
        n_estimators=600, num_leaves=31, max_depth=5, learning_rate=0.03,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=40, reg_lambda=2.0, reg_alpha=0.5,
        monotone_constraints=mc, random_state=seed, n_jobs=-1, verbose=-1,
    )


def make_catboost(columns, seed=0, monotone=True):
    from catboost import CatBoostClassifier
    mc = monotone_vector(columns) if monotone else None
    return CatBoostClassifier(
        iterations=500, depth=5, learning_rate=0.04, l2_leaf_reg=4.0,
        loss_function="Logloss", random_seed=seed, verbose=0,
        monotone_constraints=mc, allow_writing_files=False,
    )


def make_hgb(columns, seed=0, monotone=True):
    from sklearn.ensemble import HistGradientBoostingClassifier
    mc = monotone_vector(columns) if monotone else None
    return HistGradientBoostingClassifier(
        max_iter=500, max_depth=4, learning_rate=0.04, l2_regularization=2.0,
        min_samples_leaf=40, monotonic_cst=mc, random_state=seed,
    )


def make_logistic(columns, seed=0, monotone=True):
    """Interpretable baseline: median-impute + scale + L2 logistic. No NaN/monotone."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pre = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(C=0.5, max_iter=2000, random_state=seed))])


BACKBONES = {
    "xgb": make_xgb, "lgbm": make_lgbm, "catboost": make_catboost,
    "hgb": make_hgb, "logistic": make_logistic,
}


def make_backbone(name: str, columns: list[str], seed: int = 0, monotone: bool = True):
    return BACKBONES[name](columns, seed=seed, monotone=monotone)


def proba(model, X) -> np.ndarray:
    """P(default) from any backbone."""
    return model.predict_proba(X)[:, 1]
