"""Write the A/B/C submission files and provide an always-valid dummy fallback.

The golden rule of this hackathon: NEVER be in a state where you cannot submit.
`build_dummy()` writes three files into out/ that pass validate_submission.py,
so from minute one there is a submittable baseline. Real models overwrite these
piece by piece via write_a / write_b / write_c.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from . import paths


def _safe_to_csv(d: pd.DataFrame, path, retries: int = 6, delay: float = 0.7) -> None:
    """Write atomically (tmp -> replace) with retries, so a transient OneDrive
    lock self-heals and a hard lock (file open in Excel) gives a clear message."""
    path = path if hasattr(path, "with_suffix") else paths.OUT_DIR / path
    tmp = path.with_suffix(path.suffix + ".tmp")
    d.to_csv(tmp, index=False)
    for i in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == retries - 1:
                raise PermissionError(
                    f"Cannot write '{path.name}' — it is open in another program "
                    f"(close it in Excel/preview and re-run). Wrote '{tmp.name}' instead."
                )
            time.sleep(delay)

# Column orders the scorer expects.
COLUMNS_A = ["applicant_id", "decision", "predicted_pd", "pd_lower_90", "pd_upper_90"]
COLUMNS_B = ["cohort_week", "loan_age_weeks", "cumulative_default_rate", "cdr_lower_90", "cdr_upper_90"]
COLUMNS_C = ["query_id", "predicted_pd_cf", "pd_cf_lower_90", "pd_cf_upper_90"]


def _read_ids(path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def expected_applicant_ids() -> list[str]:
    return _read_ids(paths.APPLICANT_IDS)


def expected_query_ids() -> list[str]:
    return _read_ids(paths.QUERY_IDS)


def _clip01(x) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def write_a(df: pd.DataFrame, out_dir=paths.OUT_DIR) -> None:
    """df must have COLUMNS_A. Enforces ranges + interval order before writing."""
    d = df.loc[:, COLUMNS_A].copy()
    for c in ("predicted_pd", "pd_lower_90", "pd_upper_90"):
        d[c] = _clip01(d[c])
    d["pd_lower_90"] = np.minimum(d["pd_lower_90"], d["predicted_pd"])
    d["pd_upper_90"] = np.maximum(d["pd_upper_90"], d["predicted_pd"])
    d["decision"] = d["decision"].astype(int)
    out_dir.mkdir(parents=True, exist_ok=True)
    _safe_to_csv(d, out_dir / paths.FILE_A)


def write_b(df: pd.DataFrame, out_dir=paths.OUT_DIR) -> None:
    """df must have COLUMNS_B. Enforces range, interval order, and per-cohort
    monotone-nondecreasing cumulative_default_rate before writing."""
    d = df.loc[:, COLUMNS_B].copy()
    d["cohort_week"] = d["cohort_week"].astype(int)
    d["loan_age_weeks"] = d["loan_age_weeks"].astype(int)
    for c in ("cumulative_default_rate", "cdr_lower_90", "cdr_upper_90"):
        d[c] = _clip01(d[c])
    d = d.sort_values(["cohort_week", "loan_age_weeks"]).reset_index(drop=True)
    # enforce monotone non-decreasing within each cohort
    d["cumulative_default_rate"] = d.groupby("cohort_week")["cumulative_default_rate"].cummax()
    d["cdr_lower_90"] = np.minimum(d["cdr_lower_90"], d["cumulative_default_rate"])
    d["cdr_upper_90"] = np.maximum(d["cdr_upper_90"], d["cumulative_default_rate"])
    out_dir.mkdir(parents=True, exist_ok=True)
    _safe_to_csv(d, out_dir / paths.FILE_B)


def write_c(df: pd.DataFrame, out_dir=paths.OUT_DIR) -> None:
    """df must have COLUMNS_C. Enforces ranges + interval order before writing."""
    d = df.loc[:, COLUMNS_C].copy()
    for c in ("predicted_pd_cf", "pd_cf_lower_90", "pd_cf_upper_90"):
        d[c] = _clip01(d[c])
    d["pd_cf_lower_90"] = np.minimum(d["pd_cf_lower_90"], d["predicted_pd_cf"])
    d["pd_cf_upper_90"] = np.maximum(d["pd_cf_upper_90"], d["predicted_pd_cf"])
    out_dir.mkdir(parents=True, exist_ok=True)
    _safe_to_csv(d, out_dir / paths.FILE_C)


def build_dummy(out_dir=paths.OUT_DIR, pd_point: float = 0.18) -> None:
    """Write a complete, validator-passing baseline submission into out/.

    A: approve everyone at a flat PD. B: a gentle increasing curve. C: flat PD.
    Intervals are a fixed +/- band, clipped + ordered. Replace per deliverable
    as real models come online.
    """
    aids = expected_applicant_ids()
    qids = expected_query_ids()
    band = 0.10

    a = pd.DataFrame({
        "applicant_id": aids,
        "decision": 1,
        "predicted_pd": pd_point,
        "pd_lower_90": pd_point - band,
        "pd_upper_90": pd_point + band,
    })
    write_a(a, out_dir)

    n = int(__import__("json").loads(paths.MANIFEST.read_text())["n_cohort_weeks"])
    rows = []
    for w in range(1, n + 1):
        for age in range(1, n + 1):
            cdr = pd_point * age / n  # 0 -> pd_point across ages, monotone
            rows.append((w, age, cdr, max(0.0, cdr - band), cdr + band))
    b = pd.DataFrame(rows, columns=COLUMNS_B)
    write_b(b, out_dir)

    c = pd.DataFrame({
        "query_id": qids,
        "predicted_pd_cf": pd_point,
        "pd_cf_lower_90": pd_point - band,
        "pd_cf_upper_90": pd_point + band,
    })
    write_c(c, out_dir)
    print(f"dummy submission written to {out_dir}  "
          f"(A:{len(a)} rows, B:{len(b)} rows, C:{len(c)} rows)")


if __name__ == "__main__":
    build_dummy()
