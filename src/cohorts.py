"""Assign applications to origination cohort weeks (1..13) for Deliverable B.

cohort_week_definitions.csv maps each week to a [start_date, end_date] range;
validation+test timestamps fall exactly within the 13-week window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths

N_WEEKS = 13
WEEK_DAYS = 7


def cohort_table() -> pd.DataFrame:
    t = pd.read_csv(paths.COHORT_DEFS, parse_dates=["start_date", "end_date"])
    return t.sort_values("cohort_week").reset_index(drop=True)


def assign_cohort(df: pd.DataFrame) -> pd.Series:
    """Return cohort_week (1..13) for each row from application_timestamp.

    Timestamps before week 1 clamp to 1; after week 13 clamp to 13 (defensive).
    """
    ts = pd.to_datetime(df["application_timestamp"])
    tbl = cohort_table()
    week = pd.Series(np.nan, index=df.index, dtype="float")
    for _, r in tbl.iterrows():
        m = (ts >= r["start_date"]) & (ts <= r["end_date"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        week[m] = int(r["cohort_week"])
    # clamp any stragglers
    week = week.fillna(np.clip((ts - tbl["start_date"].iloc[0]).dt.days // WEEK_DAYS + 1, 1, N_WEEKS))
    return week.astype(int)


# Representative default day for each default-week index 1..13, from training data.
# (weeks 10-12 have no support; defaults are during the term or at the day-90 window.)
def week_to_day_bounds() -> np.ndarray:
    """Upper day bound (7a) for loan-age week a = 1..13 -> day 7,14,...,91."""
    return np.arange(1, N_WEEKS + 1) * WEEK_DAYS
