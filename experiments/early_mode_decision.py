"""Does targeting the COSTLY mode (early default) beat the diluted any-default
decision on VALIDATION P&L (post-drift, the real test)?  Decision:
  E[NPV] = (1 - p_early)*npv_repaid + p_early * sum_k w_early_k * npv_default(day_k)
(day-90 defaults ~ repaid, so they fall in the '1 - p_early' good bucket)."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from src import data, features, npv, survival

tr, va = data.load_train(), data.load_validation()
L = tr[tr.default_flag.notna()].copy(); V = va[va.default_flag.notna()].copy()
rec = survival.recovery_rate_from(tr)
R = V.requested_amount.values.astype(float)
dfl = (V.default_flag == 1).values; tstar = V.days_to_default.values
realized = npv.realized_npv(R, dfl, tstar, rec)

y_any = L.default_flag.astype(int).values
y_early = ((L.default_flag == 1) & (L.days_to_default <= 60)).astype(int).values
yv_any = V.default_flag.astype(int).values
yv_early = ((V.default_flag == 1) & (V.days_to_default <= 60)).astype(int).values

Xtr, fs = features.build_features(L); Xva, _ = features.build_features(V, fs)
from src.models import make_backbone, proba
def _lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def fit_pred(yt, blend=False):
    m = make_backbone("logistic", fs.columns); m.fit(Xtr, yt); pl = proba(m, Xva)
    if not blend:
        return pl
    h = make_backbone("hgb", fs.columns); h.fit(Xtr, yt); ph = proba(h, Xva)
    return 1 / (1 + np.exp(-(_lg(pl) + _lg(ph)) / 2))      # logit-avg blend
raw_any = fit_pred(y_any); raw_early = fit_pred(y_early)
blend_any = fit_pred(y_any, blend=True); blend_early = fit_pred(y_early, blend=True)
print("VAL AUC (does early-separability survive the drift?):")
print(f"  P(any default)   logistic {roc_auc_score(yv_any, raw_any):.4f}  blend {roc_auc_score(yv_any, blend_any):.4f}")
print(f"  P(early default) logistic {roc_auc_score(yv_early, raw_early):.4f}  blend {roc_auc_score(yv_early, blend_early):.4f}")

# timing laws from train defaulters
d = L[L.default_flag == 1]
wk = np.ceil(d.days_to_default.values / 7).clip(1, 13).astype(int)
def timing(mask):
    w = np.bincount(wk[mask], minlength=14)[1:14].astype(float); w = w / w.sum()
    dpw = np.array([d.days_to_default.values[(wk == k) & mask].mean() if ((wk == k) & mask).any()
                    else 7 * k - 3 for k in range(1, 14)])
    return w, dpw
w_all, dpw_all = timing(np.ones(len(d), bool))
w_early, dpw_early = timing(d.days_to_default.values <= 60)

def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def cal(pv, yv):
    cp = np.zeros(len(pv))
    for a, b in KFold(2, shuffle=True, random_state=0).split(pv):
        c = LogisticRegression().fit(logit(pv[a]).reshape(-1, 1), yv[a])
        cp[b] = c.predict_proba(logit(pv[b]).reshape(-1, 1))[:, 1]
    return cp
def pnl(enpv):
    appr = enpv > 0; return appr.mean(), realized[appr].sum()

# A) current: any-PD x full timing
cp_any = cal(raw_any, yv_any)
e_any = npv.expected_npv_weekly(R, cp_any[:, None] * w_all[None, :], dpw_all, rec)
ar0, pnl0 = pnl(e_any)
# B) early-mode: p_early x early timing (good bucket = repaid + day90)
cp_e = cal(raw_early, yv_early)
e_early = npv.expected_npv_weekly(R, cp_e[:, None] * w_early[None, :], dpw_early, rec)
ar1, pnl1 = pnl(e_early)
# C) average the two E[NPV] decisions
ar2, pnl2 = pnl((e_any + e_early) / 2)
# D) MAX config: blend(log+hgb) on both targets, average their E[NPV]
cpb_any = cal(blend_any, yv_any); cpb_e = cal(blend_early, yv_early)
eb_any = npv.expected_npv_weekly(R, cpb_any[:, None] * w_all[None, :], dpw_all, rec)
eb_e = npv.expected_npv_weekly(R, cpb_e[:, None] * w_early[None, :], dpw_early, rec)
ar3, pnl3 = pnl((eb_any + eb_e) / 2)
ar4, pnl4 = pnl(eb_e)

print("\nVAL P&L by decision basis (n=2,551, calibrated, E[NPV]>0):")
print(f"  A current  any-PD x full-timing       approve {ar0:.3f}   ${pnl0/1e3:,.0f}K")
print(f"  B early-mode  P(early) x early-timing  approve {ar1:.3f}   ${pnl1/1e3:,.0f}K   delta ${ (pnl1-pnl0)/1e3:+,.0f}K")
print(f"  C avg(A,B) E[NPV]                       approve {ar2:.3f}   ${pnl2/1e3:,.0f}K   delta ${ (pnl2-pnl0)/1e3:+,.0f}K")
print(f"  D blend(log+hgb) early-only            approve {ar4:.3f}   ${pnl4/1e3:,.0f}K   delta ${ (pnl4-pnl0)/1e3:+,.0f}K")
print(f"  E MAX: blend on both, avg E[NPV]       approve {ar3:.3f}   ${pnl3/1e3:,.0f}K   delta ${ (pnl3-pnl0)/1e3:+,.0f}K")
