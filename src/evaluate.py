"""Validation metrics: PD quality + realized policy P&L (mirrors S_P&L).

Validation outcomes exist only for prior-approved val loans (selection again), so
absolute NPV is optimistic — but it is a sound *relative* yardstick for choosing
between models/policies, which is what we use it for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import npv


def pd_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "mean_pred": float(p.mean()),
        "calib_gap": float(p.mean() - y.mean()),   # +ve => over-predicts default
    }


def trajectory_metrics(model, df_labeled: pd.DataFrame, n_weeks: int = 13) -> dict:
    """Measure Deliverable-B accuracy on labeled data (the bucket we never tested).

    Compares the model's mean predicted cumulative-default curve against the
    realized curve over `df_labeled`, plus a timing-only error among defaulters.
    """
    y = df_labeled["default_flag"].fillna(0).astype(int).to_numpy()
    wk = np.ceil(df_labeled["days_to_default"].to_numpy() / 7.0)
    wk = np.where(np.isfinite(wk), np.clip(wk, 1, n_weeks), 0).astype(int)  # 0 = no default

    ages = np.arange(1, n_weeks + 1)
    realized_cif = np.array([np.mean((y == 1) & (wk <= a) & (wk >= 1)) for a in ages])
    pred_cif = model.predict_cif(df_labeled).mean(axis=0)
    traj_mae = float(np.mean(np.abs(pred_cif - realized_cif)))

    # timing-only: week distribution among realized defaulters vs predicted
    deff = y == 1
    real_w = np.zeros(n_weeks)
    for k in wk[deff]:
        real_w[k - 1] += 1
    real_w = real_w / max(real_w.sum(), 1)
    pred_w = model.predict_week_probs(df_labeled)[deff].mean(axis=0)
    timing_mae = float(np.mean(np.abs(pred_w - real_w)))

    return {"traj_mae": traj_mae, "timing_mae": timing_mae,
            "final_pred": float(pred_cif[-1]), "final_real": float(realized_cif[-1]),
            "pred_cif": pred_cif, "realized_cif": realized_cif}


def realized_npv(df: pd.DataFrame) -> np.ndarray:
    """Per-loan realized NPV from observed outcomes (needs labels)."""
    R = df["requested_amount"].to_numpy(dtype=float)
    defaulted = df["default_flag"].fillna(0).astype(int).to_numpy()
    t = df["days_to_default"].to_numpy(dtype=float)
    rec = df["final_recovered_amount"].fillna(0).to_numpy(dtype=float)
    return npv.realized_npv(R, defaulted, t, rec)


def policy_pnl(df_labeled: pd.DataFrame, decision: np.ndarray) -> dict:
    """Realized P&L of approving the rows where decision==1 (others contribute 0)."""
    rnpv = realized_npv(df_labeled)
    d = np.asarray(decision).astype(int)
    approved = d == 1
    return {
        "approve_rate": float(approved.mean()),
        "total_pnl": float(rnpv[approved].sum()),
        "mean_pnl_per_approved": float(rnpv[approved].mean()) if approved.any() else 0.0,
        "n_approved": int(approved.sum()),
    }


def tune_buffer(df_labeled: pd.DataFrame, e_npv: np.ndarray, grid=None) -> dict:
    """Pick approval buffer tau (approve iff E[NPV] > tau) maximizing realized P&L.

    The labeled book is optimistic (selection), so the true break-even sits a bit
    above 0; tuning tau on validation calibrates the decision to actual profit.
    To avoid over-fitting the sweep, ties within 0.5% of the max prefer the
    *smaller* tau (approve more) and tau=0 if it is within tolerance.
    """
    rnpv = realized_npv(df_labeled)
    if grid is None:
        # tau >= 0 only: approving E[NPV] < 0 loans lowers EXPECTED P&L (a val-set
        # mirage on the optimistic approved-only labels). Never bet on -EV spots.
        grid = np.unique(np.concatenate([[0.0], np.linspace(0, 2500, 51)]))
    pnls = np.array([rnpv[e_npv > t].sum() for t in grid])
    best = pnls.max()
    ok = grid[pnls >= best - 0.005 * abs(best)]
    tau = 0.0 if (np.abs(ok).min() == 0.0 or (ok.min() <= 0 <= ok.max())) else ok[np.argmin(np.abs(ok))]
    return {"tau": float(tau), "pnl_at_tau": float(rnpv[e_npv > tau].sum()),
            "pnl_at_0": float(rnpv[e_npv > 0].sum()),
            "approve_rate": float((e_npv > tau).mean()),
            "grid": grid, "pnls": pnls}


def compare_policies(df_labeled: pd.DataFrame, e_npv: np.ndarray, pd_point: np.ndarray) -> pd.DataFrame:
    """Benchmark the NPV-sign policy against approve-all and PD-threshold policies."""
    rows = []
    rows.append(("NPV>0 (ours)", policy_pnl(df_labeled, (e_npv > 0).astype(int))))
    rows.append(("approve-all", policy_pnl(df_labeled, np.ones(len(df_labeled), int))))
    for tau in (0.2, 0.3, 0.5):
        rows.append((f"PD<{tau}", policy_pnl(df_labeled, (pd_point < tau).astype(int))))
    out = pd.DataFrame({name: m for name, m in rows}).T
    return out[["approve_rate", "n_approved", "total_pnl", "mean_pnl_per_approved"]]
