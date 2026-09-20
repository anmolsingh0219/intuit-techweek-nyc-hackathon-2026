"""Weight-of-Evidence encoder (monotone scorecard transform).

Replaces raw/engineered columns with their WOE = ln(%non-default / %default),
using supervised monotonic bins (quantile -> pool-adjacent-violators) and a
dedicated MISSING bin. Output is a clean, low-collinearity, drift-robust matrix
for the logistic backbone, and a coefficient table that *is* a credit scorecard.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Pruned, de-correlated feature set chosen from the IV / PSI / redundancy analysis.
NUMERIC = [
    "invoice_payment_delinquency_rate", "aggregate_credit_utilization",
    "observed_cash_balance_p10", "requested_amount_to_observed_revenue",
    "requested_amount", "payroll_regularity_score", "observed_revenue_volatility",
    "existing_debt_obligations", "observed_overdraft_count_3mo",
    "observed_monthly_revenue_avg_3mo", "bookkeeping_recency_days", "vintage_years",
    "days_since_last_external_decline", "days_since_last_inquiry_elsewhere",
]
ENG_NUMERIC = ["stated_vs_observed_rev", "debt_service_ratio"]  # from features._engineer
CATEGORICAL = ["owner_personal_credit_band", "employee_count_bucket", "sector"]

MISSING = "·MISSING·"
OTHER = "·OTHER·"


def _woe_map(bad, good, eps=0.5):
    bad, good = np.asarray(bad, float), np.asarray(good, float)
    tb, tg, k = bad.sum(), good.sum(), len(bad)
    db = (bad + eps) / (tb + eps * k)
    dg = (good + eps) / (tg + eps * k)
    woe = np.log(dg / db)
    iv = float(((dg - db) * woe).sum())
    return woe, iv


def _monotone_edges(x, y, q=10, min_frac=0.05):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    xv, yv = x[m], np.asarray(y)[m]
    if len(np.unique(xv)) < 3:
        return None
    qs = np.unique(np.nanquantile(xv, np.linspace(0, 1, q + 1)))
    if len(qs) < 3:
        return None
    edges = qs.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    N = len(xv)

    def rates(ed):
        idx = np.digitize(xv, ed[1:-1])
        g = pd.DataFrame({"i": idx, "y": yv}).groupby("i")["y"]
        return g.size(), g.mean()

    changed = True
    while changed and len(edges) > 3:
        changed = False
        sz, _ = rates(edges)
        nb = len(edges) - 1
        for i in range(nb):
            if sz.get(i, 0) < min_frac * N:
                edges = np.delete(edges, i + 1 if i < nb - 1 else i)
                changed = True
                break
    sz, rt = rates(edges)
    r = np.array([rt.get(i, np.nan) for i in range(len(edges) - 1)])
    r = r[np.isfinite(r)]
    direction = np.sign(np.corrcoef(np.arange(len(r)), r)[0, 1]) if (len(r) > 1 and np.ptp(r) > 0) else 1.0
    direction = direction or 1.0
    guard = 0
    while len(edges) > 3 and guard < 60:
        guard += 1
        _, rt = rates(edges)
        r = np.array([rt.get(i, np.nan) for i in range(len(edges) - 1)])
        bad = np.where(np.diff(r) * direction < 0)[0]
        if len(bad) == 0:
            break
        edges = np.delete(edges, bad[0] + 1)
    return edges


@dataclass
class WOEEncoder:
    numeric: list = field(default_factory=lambda: NUMERIC + ENG_NUMERIC)
    categorical: list = field(default_factory=lambda: list(CATEGORICAL))
    min_frac: float = 0.05
    num_bins_: dict = field(default_factory=dict)   # col -> (edges, woe_per_bin, missing_woe)
    cat_bins_: dict = field(default_factory=dict)   # col -> ({level: woe}, missing_woe, other_woe)
    iv_: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, y):
        y = np.asarray(y).astype(int)
        for c in self.numeric:
            if c not in df:
                continue
            x = pd.to_numeric(df[c], errors="coerce").values
            edges = _monotone_edges(x, y, min_frac=self.min_frac)
            if edges is None:
                continue
            idx = np.digitize(x, edges[1:-1])
            miss = ~np.isfinite(x)
            nb = len(edges) - 1
            bad = np.zeros(nb + 1); good = np.zeros(nb + 1)  # last slot = MISSING
            for b in range(nb):
                sel = (idx == b) & ~miss
                bad[b], good[b] = y[sel].sum(), (~y[sel].astype(bool)).sum()
            bad[nb], good[nb] = y[miss].sum(), (~y[miss].astype(bool)).sum()
            woe, iv = _woe_map(bad, good)
            self.num_bins_[c] = (edges, woe[:nb], woe[nb])
            self.iv_[c] = iv
        for c in self.categorical:
            if c not in df:
                continue
            s = df[c].astype("object").where(df[c].notna(), MISSING)
            vc = s.value_counts()
            rare = vc[vc < max(self.min_frac * len(s), 20)].index
            s = s.where(~s.isin(rare), OTHER)
            levels = list(pd.unique(s))
            bad = np.array([y[(s == lv).values].sum() for lv in levels], float)
            good = np.array([(~y[(s == lv).values].astype(bool)).sum() for lv in levels], float)
            woe, iv = _woe_map(bad, good)
            wmap = {lv: w for lv, w in zip(levels, woe)}
            self.cat_bins_[c] = (wmap, wmap.get(MISSING, 0.0), wmap.get(OTHER, 0.0))
            self.iv_[c] = iv
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c, (edges, woe, mwoe) in self.num_bins_.items():
            x = pd.to_numeric(df.get(c), errors="coerce").values if c in df else np.full(len(df), np.nan)
            idx = np.digitize(x, edges[1:-1])
            v = np.where(np.isfinite(x), woe[np.clip(idx, 0, len(woe) - 1)], mwoe)
            out[c + "_woe"] = v
        for c, (wmap, mwoe, owoe) in self.cat_bins_.items():
            s = df[c].astype("object").where(df[c].notna(), MISSING) if c in df else pd.Series(MISSING, index=df.index)
            out[c + "_woe"] = s.map(lambda lv: wmap.get(lv, owoe)).astype(float).values
        return pd.DataFrame(out, index=df.index)

    def fit_transform(self, df, y):
        return self.fit(df, y).transform(df)

    def iv_table(self) -> pd.DataFrame:
        return (pd.Series(self.iv_, name="IV").sort_values(ascending=False)
                .round(4).to_frame())
