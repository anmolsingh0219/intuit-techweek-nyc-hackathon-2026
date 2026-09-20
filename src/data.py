"""Data loading + authoritative schema for the SMB Underwriting Challenge.

Single source of truth for: where the frames come from, which columns are
features vs outcomes vs ids, which are intervenable, and how to build the
survival label (event, duration, censoring) from the raw outcome columns.

Nothing here trains a model — it just gives every downstream module the same
clean, consistent view of the data.
"""
from __future__ import annotations

import json
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd

from . import paths

# --------------------------------------------------------------------------- #
# Raw frame loading
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=None)
def _load_zip_member(name: str) -> pd.DataFrame:
    with zipfile.ZipFile(paths.DATASET_ZIP) as z:
        with z.open(name) as f:
            return pd.read_csv(f)


def load_train() -> pd.DataFrame:
    return _load_zip_member("train.csv").copy()


def load_validation() -> pd.DataFrame:
    return _load_zip_member("validation.csv").copy()


def load_test() -> pd.DataFrame:
    return _load_zip_member("test.csv").copy()


def load_all() -> dict[str, pd.DataFrame]:
    return {"train": load_train(), "validation": load_validation(), "test": load_test()}


# --------------------------------------------------------------------------- #
# Schema (parsed from data_dictionary.csv — authoritative, not hardcoded)
# --------------------------------------------------------------------------- #

ID_COLS = ("business_id", "applicant_id")
# Outcome columns must NEVER be used as features (leakage).
OUTCOME_COLS = (
    "default_flag",
    "days_to_default",
    "days_to_full_repayment",
    "repayment_status",
    "final_recovered_amount",
    "observation_status",
)


@lru_cache(maxsize=1)
def data_dictionary() -> pd.DataFrame:
    return pd.read_csv(paths.DATA_DICTIONARY)


@lru_cache(maxsize=1)
def manifest() -> dict:
    return json.loads(paths.MANIFEST.read_text())


def column_groups() -> dict[str, list[str]]:
    """field group -> list of columns, straight from the data dictionary."""
    dd = data_dictionary()
    return {g: sorted(sub["field"].tolist()) for g, sub in dd.groupby("group")}


def intervenable_features() -> list[str]:
    """Features the challenge lets you intervene on (Deliverable C)."""
    dd = data_dictionary()
    col = dd["intervenable"].astype(str).str.strip().str.lower()
    return dd.loc[col.eq("true"), "field"].tolist()


def feature_columns() -> list[str]:
    """Columns usable as model inputs: everything except ids + outcomes.

    NOTE: application_timestamp is kept here (it is needed to derive cohort_week
    and time-based features); drop or transform it inside the feature pipeline,
    not here, so the raw list stays honest.
    """
    dd = data_dictionary()
    drop = set(ID_COLS) | set(OUTCOME_COLS)
    return [c for c in dd["field"].tolist() if c not in drop]


def nullable_feature_columns() -> list[str]:
    """Feature columns whose null is structural (MNAR) — get is_missing flags.

    Heuristic: the data dictionary notes say "Null if ..." / "null when ...".
    """
    dd = data_dictionary()
    feats = set(feature_columns())
    notes = dd["notes"].astype(str).str.lower()
    mask = notes.str.contains("null") & dd["field"].isin(feats)
    return dd.loc[mask, "field"].tolist()


# --------------------------------------------------------------------------- #
# Survival label construction
# --------------------------------------------------------------------------- #

DEFAULT_WINDOW_DAYS = 90  # the README "default window": balance>0 at day 90 => default


def build_survival_label(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-row (event, duration, has_label) for time-to-default modeling.

    Conventions (right-censored survival):
      - event = 1, duration = days_to_default   for observed defaults.
      - event = 0, duration = days_to_full_repayment   for paid-in-full loans
        (censored at the day they fully repaid — they never defaulted).
      - has_label = False for rows with no observed outcome (prior-declined or
        immature). These carry NO event/duration and must be excluded from the
        survival fit (but ARE the population for selection-bias correction).

    Only prior-approved + matured loans have labels; this is the selective-labels
    structure at the heart of the challenge.
    """
    out = pd.DataFrame(index=df.index)
    has_label = df["default_flag"].notna()
    defaulted = df["default_flag"].fillna(0).astype(bool) & has_label

    duration = np.where(
        defaulted,
        df["days_to_default"],
        df["days_to_full_repayment"],
    ).astype(float)

    out["has_label"] = has_label.to_numpy()
    out["event"] = defaulted.astype(int).to_numpy()
    out["duration"] = duration
    # Rows with a label but missing both day fields (shouldn't happen) -> drop.
    out.loc[out["has_label"] & ~np.isfinite(out["duration"]), "has_label"] = False
    return out


def approval_indicator(df: pd.DataFrame) -> pd.Series:
    """1 if the PRIOR underwriter approved (=> outcome is observed), else 0.

    This is the selection mechanism: labels exist only where prior_decision == 1.
    Used to model the propensity of approval for IPW / selection-bias correction.
    """
    return df["prior_decision"].astype("float").fillna(0).astype(int)


if __name__ == "__main__":  # quick smoke test
    frames = load_all()
    for k, v in frames.items():
        print(f"{k:11s} {v.shape}")
    print("\ngroups:", {g: len(c) for g, c in column_groups().items()})
    print("\nintervenable:", intervenable_features())
    print("\nnullable (MNAR flags):", nullable_feature_columns())
    lab = build_survival_label(frames["train"])
    print("\ntrain labels: has_label", int(lab["has_label"].sum()),
          "events", int(lab.loc[lab.has_label, "event"].sum()))
