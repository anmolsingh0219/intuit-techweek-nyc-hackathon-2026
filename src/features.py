"""Feature pipeline (Phase 1.5: proper categorical encoding + interactions).

Design choices (defended in the writeup):
  - Tree models handle NaN natively, so we do NOT impute — we keep NaN and ADD
    explicit `__missing` flags so the model can use the *fact* of missingness
    (MNAR signal). The logistic backbone median-imputes inside its own pipeline.
  - NOMINAL categoricals (sector, geography, intended-use, channel) carry no
    order, so feeding their integer codes to a LINEAR model is wrong ("code 5 ==
    5x code 1"). We ONE-HOT them. Low cardinality (<=5 levels) makes this cheap
    and leak-free (unlike target/WOE encoding).
  - ORDINAL categoricals (employee_count_bucket, owner_personal_credit_band) keep
    their numeric order.
  - A small, economically-motivated set of interaction + squared terms gives the
    linear model the non-linearity that trees get for free.
  - DROP `application_timestamp` (cohort only), `prior_decision` /
    `prior_approved_amount` (constant / post-selection among labeled rows).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data

EXTRA_DROP = ("application_timestamp", "prior_decision", "prior_approved_amount")
NOMINAL = ("sector", "geography_region", "intended_use_of_funds", "application_channel")


@dataclass
class FeatureState:
    """Fitted feature spec so val/test get exactly the same columns, in order."""
    columns: list[str]
    indicator_cols: list[str] = field(default_factory=list)   # flags + one-hot -> fill NaN with 0


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio, interaction, squared, and credit-risk domain features. inf -> NaN."""
    e = pd.DataFrame(index=df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        rev = df["stated_annual_revenue"].replace(0, np.nan)
        util = df["aggregate_credit_utilization"]
        score = df["prior_underwriter_score"]
        delinq = df["invoice_payment_delinquency_rate"]
        inq = df["recent_inquiries_count_6mo"]
        amt = df["requested_amount"]
        obs_m = df.get("observed_monthly_revenue_avg_3mo", pd.Series(np.nan, index=df.index))
        obs_daily = obs_m / 30.0
        cash = df.get("observed_cash_balance_p10", pd.Series(np.nan, index=df.index))
        debt = df["existing_debt_obligations"]

        # --- leverage / affordability (core credit-risk) ---
        e["req_to_stated_rev"] = amt / rev
        e["debt_to_stated_rev"] = debt / rev
        e["stated_vs_observed_rev"] = (df["stated_annual_revenue"] / 12.0) / obs_m.replace(0, np.nan)
        e["loan_to_observed_annual_rev"] = amt / (obs_m.replace(0, np.nan) * 12.0)
        # daily ACH draw (fixed product math) vs daily revenue -> can they service it?
        daily_draw = amt * (1.0 + 0.35 * 60 / 365) / 60.0
        e["draw_to_daily_rev"] = daily_draw / obs_daily.replace(0, np.nan)
        e["debt_service_ratio"] = debt / obs_m.replace(0, np.nan)
        # cash runway (days of cash at current burn); negative => already underwater
        e["cash_runway_days"] = cash / obs_daily.replace(0, np.nan)
        e["draw_to_cash_buffer"] = amt / cash.replace(0, np.nan)

        # --- behaviour / bureau intensity ---
        pl = df["prior_loans_count"].replace(0, np.nan)
        e["prior_default_rate"] = df["prior_loans_default_count"] / pl
        e["inquiry_intensity"] = inq + df["multi_lender_inquiry_count_30d"]
        e["overdraft_per_month"] = df.get("observed_overdraft_count_3mo", np.nan) / 3.0
        e["revenue_signal"] = df.get("observed_revenue_trend_3mo", 0) / (
            df.get("observed_revenue_volatility", pd.Series(np.nan, index=df.index)).replace(0, np.nan))

        # --- interactions (risk drivers that compound) ---
        e["util_x_inquiries"] = util * inq
        e["leverage_x_delinq"] = e["req_to_stated_rev"] * delinq
        e["score_x_util"] = score * util
        e["util_x_band"] = util * df["owner_personal_credit_band"]
        e["affordability_x_delinq"] = e["draw_to_daily_rev"] * delinq
        # --- curvature for the linear model ---
        e["score_sq"] = score ** 2
        e["util_sq"] = util ** 2
    return e.replace([np.inf, -np.inf], np.nan)


def build_features(df: pd.DataFrame, state: FeatureState | None = None) -> tuple[pd.DataFrame, FeatureState]:
    """Return (X, state). Pass the train-fitted `state` for val/test consistency."""
    base_cols = [c for c in data.feature_columns()
                 if c not in EXTRA_DROP and c not in NOMINAL]
    base = df[base_cols].apply(pd.to_numeric, errors="coerce")

    # MNAR flags
    nullable = [c for c in data.nullable_feature_columns() if c in base_cols]
    flags = pd.DataFrame({f"{c}__missing": df[c].isna().astype(int) for c in nullable}, index=df.index)

    # one-hot nominal categoricals
    nom = df[list(NOMINAL)].apply(lambda s: s.astype("Int64"))
    onehot = pd.get_dummies(nom, columns=list(NOMINAL), prefix=list(NOMINAL),
                            prefix_sep="=", dummy_na=False).astype(int)

    eng = _engineer(df)
    X = pd.concat([base, flags, onehot, eng], axis=1)
    indicator = list(flags.columns) + list(onehot.columns)

    if state is None:
        state = FeatureState(columns=list(X.columns), indicator_cols=indicator)
    else:
        X = X.reindex(columns=state.columns)
        # absent one-hot/flag columns mean "0", not "missing"
        present = [c for c in state.indicator_cols if c in X.columns]
        X[present] = X[present].fillna(0)
    return X, state


if __name__ == "__main__":
    tr = data.load_train()
    X, st = build_features(tr)
    print("feature matrix:", X.shape, "| indicator cols:", len(st.indicator_cols))
    print("one-hot:", [c for c in st.columns if "=" in c])
