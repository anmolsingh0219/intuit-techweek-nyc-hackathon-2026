"""Shared model+calibration assembly (used by Phase 2/3 so they agree exactly).

Returns the Phase-1-selected risk model (bagged logistic), a PD calibrator chosen
by held-out Brier, an honestly-tuned PD-interval (k, floor), and the NPV buffer tau.
"""
from __future__ import annotations

import numpy as np

from . import calibrate, data, evaluate, npv, survival


def fit_calibrated_model(labeled, val, rec: float, backbone: str = "logistic", k_bag: int = 5):
    """-> (model, calibrator, kA dict, tau, method). `val` is labeled validation rows.

    backbone='logistic' for A/B (best ranking/P&L); 'hgb' for C counterfactuals
    (bounded tree -> credible interventional magnitudes, NaN-native).
    """
    model = survival.RiskModel(backbone=backbone, k_bag=k_bag, recovery_rate=rec).fit(labeled)
    raw_val, _ = model.predict_pd(val)
    y_val = val["default_flag"].astype(int).to_numpy()

    # choose calibrator + honest interval params on a held-out half
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(val)); h = len(val) // 2
    ca, ho = perm[:h], perm[h:]
    method, _ = calibrate.select_pd_calibrator(raw_val[ca], y_val[ca], raw_val[ho], y_val[ho])
    cal_h = calibrate.fit_pd_calibrator(raw_val[ca], y_val[ca], method)
    kA = calibrate.tune_pd_interval(calibrate.apply_calibrator(cal_h, raw_val[ho]), y_val[ho])

    cal = calibrate.fit_pd_calibrator(raw_val, y_val, method)   # final on all val

    # NPV buffer on calibrated val (conditional timing for the decision)
    model.timing_mode = "conditional"
    pi = calibrate.apply_calibrator(cal, raw_val)[:, None] * model.predict_week_probs(val)
    e_npv = npv.expected_npv_weekly(val["requested_amount"].to_numpy(float),
                                    pi, model.day_per_week, rec)
    tau = evaluate.tune_buffer(val, e_npv)["tau"]
    return model, cal, kA, tau, method
