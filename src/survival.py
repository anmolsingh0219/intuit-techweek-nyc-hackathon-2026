"""The core risk model — one spine for Deliverables A and B.

Because the labeled data is *fully observed* (every matured loan either defaults
on a known day or pays in full at day 60 — no censoring), we factor the discrete-
time default process into two estimable pieces:

    Stage 1  PD(x)            = P(default by day 90 | x)              [binary]
    Stage 2  w(k | x)         = P(default in week k | defaults, x)   [multiclass]

From these:
    pi[i,k]   = PD_i * w(k | x_i)            unconditional weekly default prob
    CIF[i,a]  = sum_{k<=a} pi[i,k]           cumulative default by week a (monotone)
    E[NPV_i]  = (1-PD_i)*npv_repaid + sum_k pi[i,k]*npv_default(day_k)

Deliverable A reads off PD_i and the NPV sign; Deliverable B aggregates CIF over
each cohort. Stage 1 is bagged (K seeds) to get a model-uncertainty band on PD.

Selection bias: labels exist only for prior-approved loans. `Propensity` models
P(approve | x) on the full train book; 1/propensity weights reweight the labeled
training distribution toward the full applicant population (covariate-shift /
IPW correction). Use of weights is a toggle chosen by validation performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from . import cohorts, data, features, models, npv

N_WEEKS = cohorts.N_WEEKS


def _xgb(seed: int, **kw) -> XGBClassifier:
    params = dict(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, tree_method="hist", eval_metric="logloss",
        random_state=seed, n_jobs=0,
    )
    params.update(kw)
    return XGBClassifier(**params)


# --------------------------------------------------------------------------- #
# Selection-bias propensity model
# --------------------------------------------------------------------------- #

@dataclass
class Propensity:
    model: XGBClassifier
    fstate: features.FeatureState
    p_mean: float

    @classmethod
    def fit(cls, train_full: pd.DataFrame) -> "Propensity":
        X, fs = features.build_features(train_full)
        y = data.approval_indicator(train_full).to_numpy()
        m = _xgb(0, n_estimators=300).fit(X, y)
        return cls(model=m, fstate=fs, p_mean=float(y.mean()))

    def approve_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = features.build_features(df, self.fstate)
        return self.model.predict_proba(X)[:, 1]

    def ipw_weights(self, df: pd.DataFrame, clip=(0.05, 0.95)) -> np.ndarray:
        """Stabilized inverse-propensity weights for labeled (approved) rows."""
        p = np.clip(self.approve_proba(df), 1e-3, 1 - 1e-3)
        w = self.p_mean / p                      # stabilized IPW
        lo, hi = np.quantile(w, clip)
        return np.clip(w, lo, hi)


# --------------------------------------------------------------------------- #
# Two-stage risk model
# --------------------------------------------------------------------------- #

@dataclass
class RiskModel:
    backbone: str = "xgb"                            # PD family (see models.BACKBONES)
    monotone: bool = True                            # apply monotonic constraints
    timing_mode: str = "conditional"                 # 'conditional' (x-dependent) or 'marginal'
    fstate: features.FeatureState = None
    pd_bag: list = field(default_factory=list)      # Stage-1 ensemble
    timing: XGBClassifier = None                     # Stage-2 multiclass
    marginal_week: np.ndarray = None                 # x-independent week distribution
    week_classes: np.ndarray = None                  # observed default weeks (1..13)
    day_per_week: np.ndarray = None                  # representative day, len N_WEEKS
    recovery_rate: float = 0.0
    k_bag: int = 5

    def fit(self, df_labeled: pd.DataFrame, sample_weight: np.ndarray | None = None) -> "RiskModel":
        X, fs = features.build_features(df_labeled)
        self.fstate = fs
        cols = fs.columns
        y = df_labeled["default_flag"].astype(int).to_numpy()
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

        # Stage 1: bagged PD via bootstrap resampling (works for any backbone).
        self.pd_bag = []
        n = len(X)
        for s in range(self.k_bag):
            rng = np.random.default_rng(100 + s)
            idx = rng.integers(0, n, n)
            m = models.make_backbone(self.backbone, cols, seed=100 + s, monotone=self.monotone)
            ws = None if w is None else w[idx]
            m.fit(X.iloc[idx], y[idx], **({"sample_weight": ws} if w is not None else {}))
            self.pd_bag.append(m)

        # Stage 2: timing among defaulters
        d = df_labeled[df_labeled["default_flag"] == 1]
        Xd, _ = features.build_features(d, fs)
        wk = np.ceil(d["days_to_default"].to_numpy() / 7.0).clip(1, N_WEEKS).astype(int)
        self.week_classes = np.array(sorted(np.unique(wk)))
        cls_index = {c: i for i, c in enumerate(self.week_classes)}
        yk = np.array([cls_index[v] for v in wk])
        wd = None if w is None else w[df_labeled["default_flag"].to_numpy() == 1]
        self.timing = _xgb(7, n_estimators=300, max_depth=4, objective="multi:softprob",
                           num_class=len(self.week_classes), eval_metric="mlogloss")
        self.timing.fit(Xd, yk, sample_weight=wd)

        # marginal (x-independent) week distribution, weighted -> length N_WEEKS
        mw = np.zeros(N_WEEKS)
        ww = np.ones(len(wk)) if wd is None else wd
        for k, weight in zip(wk, ww):
            mw[k - 1] += weight
        self.marginal_week = mw / mw.sum()

        # representative default day per week (mean within week; fallback 7k-3)
        self.day_per_week = np.array([7 * k - 3 for k in range(1, N_WEEKS + 1)], dtype=float)
        for k in range(1, N_WEEKS + 1):
            sel = wk == k
            if sel.any():
                self.day_per_week[k - 1] = float(d["days_to_default"].to_numpy()[sel].mean())
        return self

    def add_pd_backbone(self, df_labeled, backbone, k=3, sample_weight=None) -> "RiskModel":
        """Append bootstrap PD members from another family -> turns this into an
        ensemble that shares the already-fitted timing model (Option C)."""
        X, _ = features.build_features(df_labeled, self.fstate)
        y = df_labeled["default_flag"].astype(int).to_numpy()
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        n = len(X)
        for s in range(k):
            rng = np.random.default_rng(900 + s)
            idx = rng.integers(0, n, n)
            m = models.make_backbone(backbone, self.fstate.columns, seed=900 + s, monotone=self.monotone)
            ws = None if w is None else w[idx]
            m.fit(X.iloc[idx], y[idx], **({"sample_weight": ws} if w is not None else {}))
            self.pd_bag.append(m)
        return self

    # ---- predictions ----
    def predict_pd(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X, _ = features.build_features(df, self.fstate)
        preds = np.column_stack([models.proba(m, X) for m in self.pd_bag])
        return preds.mean(axis=1), preds.std(axis=1)

    def pd_bag_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """(n, k_bag) PD from each bag member — for B's model-uncertainty band."""
        X, _ = features.build_features(df, self.fstate)
        return np.column_stack([models.proba(m, X) for m in self.pd_bag])

    def predict_week_probs(self, df: pd.DataFrame) -> np.ndarray:
        """(n, N_WEEKS) P(default in week k | defaults). Zeros for unsupported weeks."""
        if self.timing_mode == "marginal":
            return np.tile(self.marginal_week, (len(df), 1))
        X, _ = features.build_features(df, self.fstate)
        p = self.timing.predict_proba(X)                       # (n, n_classes)
        full = np.zeros((len(df), N_WEEKS))
        full[:, self.week_classes - 1] = p
        s = full.sum(axis=1, keepdims=True)
        return np.divide(full, s, out=np.zeros_like(full), where=s > 0)

    def predict_pi(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (pd_point, pd_std, pi[n,N_WEEKS])."""
        pd_point, pd_std = self.predict_pd(df)
        wk = self.predict_week_probs(df)
        pi = pd_point[:, None] * wk
        return pd_point, pd_std, pi

    def predict_cif(self, df: pd.DataFrame) -> np.ndarray:
        _, _, pi = self.predict_pi(df)
        return np.cumsum(pi, axis=1)                            # (n, N_WEEKS), monotone

    def expected_npv(self, df: pd.DataFrame) -> np.ndarray:
        _, _, pi = self.predict_pi(df)
        R = df["requested_amount"].to_numpy(dtype=float)
        return npv.expected_npv_weekly(R, pi, self.day_per_week, self.recovery_rate)


def recovery_rate_from(train: pd.DataFrame) -> float:
    d = train[train["repayment_status"] == "defaulted"]
    r = (d["final_recovered_amount"] / d["requested_amount"]).replace([np.inf, -np.inf], np.nan)
    return float(r.dropna().mean())
