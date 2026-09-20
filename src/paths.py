"""Canonical paths for the project. Import these instead of hardcoding."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset"
DATASET_ZIP = DATASET_DIR / "dataset-compressed.zip"
DATA_DICTIONARY = DATASET_DIR / "data_dictionary.csv"
INTERVENTION_QUERIES = DATASET_DIR / "intervention_queries.csv"
COHORT_DEFS = DATASET_DIR / "cohort_week_definitions.csv"
B_TEMPLATE = DATASET_DIR / "submission_B_template.csv"

EXPECTED_IDS_DIR = ROOT / "expected_ids"
APPLICANT_IDS = EXPECTED_IDS_DIR / "applicant_ids.txt"
QUERY_IDS = EXPECTED_IDS_DIR / "query_ids.txt"
MANIFEST = EXPECTED_IDS_DIR / "manifest.json"

VALIDATOR = ROOT / "validate_submission.py"

OUT_DIR = ROOT / "out"          # the 4 submission files live here
REPORTS_DIR = ROOT / "reports"  # EDA + diagnostics
CACHE_DIR = ROOT / "experiments" / "_cache"

# Submission file names (exact, enforced by the scorer)
FILE_A = "submission_A_decisions.csv"
FILE_B = "submission_B_trajectory.csv"
FILE_C = "submission_C_counterfactuals.csv"
FILE_D = "submission_D_writeup.pdf"
