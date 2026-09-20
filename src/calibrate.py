"""Phase 2 — calibration & uncertainty quantification (the 0.20 S_cal bucket).

Two pieces:
  1. PD POINT calibration via isotonic regression fit on held-out validation, so
     predicted_pd is an honest probability (lower Brier, smaller calib gap). This
     also sharpens the E[NPV] decision (better-calibrated risk -> better folds).
  2. 90% INTERVALS via a one-parameter conformal inflation tuned on validation to
     hit ~90% coverage without being needlessly wide:
       - A (PD): half-width k * sqrt(p(1-p)) (Bernoulli-shaped, widest mid-range),
         k tuned so binned reliability coverage (realized local default rate inside
         the band) >= 90%, with a floor so intervals are not degenerate. Per-applicant
         true PD is unobservable from binary data, so this characterizes the
         estimable (calibration + resolution) uncertainty — stated plainly in D.
       - B (trajectory): half-width k * sqrt(cdr(1-cdr)/n_cohort), k tuned so the
         realized per-cohort/age cumulative default fraction lands in the band 90%.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


# --------------------------------------------------------------------------- #
# PD point calibration — isotonic (flexible) vs Platt (1-param, overfit-robust)
# --------------------------------------------------------------------------- #

def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class Calibrator:
    kind: str
    model: object

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 0, 1)
        if self.kind == "isotonic":
            return np.clip(self.model.predict(p), 0.0, 1.0)
        return self.model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]  # platt


def fit_pd_calibrator(p: np.ndarray, y: np.ndarray, kind: str = "isotonic") -> Calibrator:
    if kind == "isotonic":
        m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.clip(p, 0, 1), y)
    else:  # platt: logistic on the logit of the raw probability
        m = LogisticRegression(C=1e6, solver="lbfgs").fit(_logit(p).reshape(-1, 1), y)
    return Calibrator(kind, m)


def select_pd_calibrator(p_cal, y_cal, p_ho, y_ho) -> tuple[str, float]:
    """Pick isotonic vs platt by held-out Brier (lower is better)."""
    best, best_brier = "isotonic", np.inf
    for kind in ("isotonic", "platt"):
        c = fit_pd_calibrator(p_cal, y_cal, kind)
        b = brier_score_loss(y_ho, np.clip(c.predict(p_ho), 1e-6, 1 - 1e-6))
        if b < best_brier:
            best, best_brier = kind, b
    return best, float(best_brier)


def apply_calibrator(cal: Calibrator, p: np.ndarray) -> np.ndarray:
    return cal.predict(p)


def calibration_report(p: np.ndarray, y: np.ndarray) -> dict:
    return {"brier": float(brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            "mean_pred": float(p.mean()), "base_rate": float(y.mean()),
            "calib_gap": float(p.mean() - y.mean())}


# --------------------------------------------------------------------------- #
# A — PD intervals
# --------------------------------------------------------------------------- #

def tune_pd_interval(p: np.ndarray, y: np.ndarray, n_bins: int = 10, target: float = 0.90,
                     floors=(0.03, 0.04, 0.05, 0.06, 0.08, 0.10)) -> dict:
    """Pick (k, floor) minimizing mean width s.t. binned reliability coverage >= target.

    Band = p +/- max(k*sqrt(p(1-p)), floor). 'Covered' = the bin's realized default
    rate (a local estimate of the true PD) lies in the averaged band. Tuned on
    held-out data this is an honest 90% target."""
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    centers = np.array([p[b].mean() for b in bins])
    realized = np.array([y[b].mean() for b in bins])
    spread = np.sqrt(np.clip(centers * (1 - centers), 1e-9, None))
    best = None
    for floor in floors:
        for k in np.linspace(0.0, 4.0, 81):
            half = np.maximum(k * spread, floor)
            cov = float(np.mean(np.abs(realized - centers) <= half))
            width = float(np.mean(2 * half))
            if cov >= target and (best is None or width < best["width"]):
                best = {"k": float(k), "floor": float(floor), "coverage": cov, "width": width}
    if best is None:  # nothing hit target -> widest option
        floor = max(floors)
        half = np.maximum(4.0 * spread, floor)
        best = {"k": 4.0, "floor": float(floor),
                "coverage": float(np.mean(np.abs(realized - centers) <= half)),
                "width": float(np.mean(2 * half))}
    return best


def pd_intervals(p: np.ndarray, k: float, floor: float = 0.04):
    half = np.maximum(k * np.sqrt(np.clip(p * (1 - p), 0, None)), floor)
    return np.clip(p - half, 0, 1), np.clip(p + half, 0, 1)


# --------------------------------------------------------------------------- #
# B — trajectory intervals
# --------------------------------------------------------------------------- #

def tune_b_interval(pred_cdr: np.ndarray, real_cdr: np.ndarray, n_cohort: np.ndarray,
                    target: float = 0.90, floor: float = 0.01) -> dict:
    """k so >=target of (cohort,age) cells have realized CDR inside pred +/- k*se."""
    se = np.sqrt(np.clip(pred_cdr * (1 - pred_cdr), 1e-9, None) / np.clip(n_cohort, 1, None))
    valid = np.isfinite(real_cdr)
    for k in np.linspace(0.0, 4.0, 161):
        half = np.maximum(k * se, floor)
        cov = np.mean((np.abs(real_cdr - pred_cdr) <= half)[valid])
        if cov >= target:
            return {"k": float(k), "floor": floor, "coverage": float(cov)}
    half = np.maximum(4.0 * se, floor)
    return {"k": 4.0, "floor": floor, "coverage": float(np.mean((np.abs(real_cdr - pred_cdr) <= half)[valid]))}
