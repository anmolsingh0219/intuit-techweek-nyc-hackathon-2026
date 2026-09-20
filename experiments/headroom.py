"""Where is the remaining headroom? Decompose val P&L gap-to-oracle + check
within-test-window drift (Deliverable B lever)."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from src import data, npv, survival

tr, va = data.load_train(), data.load_validation()
L = tr[tr.default_flag.notna()].copy(); V = va[va.default_flag.notna()].copy()
y, yv = L.default_flag.astype(int).values, V.default_flag.astype(int).values
rec = survival.recovery_rate_from(tr)
R = V.requested_amount.values.astype(float); dfl = (V.default_flag == 1).values
tstar = V.days_to_default.values
realized = npv.realized_npv(R, dfl, tstar, rec)

# production-faithful model: logistic two-stage, conditional timing
model = survival.RiskModel("logistic", recovery_rate=rec).fit(L)
raw, _ = model.predict_pd(V)
wkp = model.predict_week_probs(V)            # (n,13) conditional timing
# 2-fold Platt calibrate
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
cp = np.zeros(len(raw))
for a, b in KFold(2, shuffle=True, random_state=0).split(raw):
    c = LogisticRegression().fit(logit(raw[a]).reshape(-1, 1), yv[a]); cp[b] = c.predict_proba(logit(raw[b]).reshape(-1, 1))[:, 1]
pi = cp[:, None] * wkp
enpv = npv.expected_npv_weekly(R, pi, model.day_per_week, rec)

# ---------- 1. P&L decomposition ----------
oracle = realized[realized > 0].sum()
appr_all = realized.sum()
appr = enpv > 0
ours = realized[appr].sum()
fa = realized[appr & (realized < 0)]          # false approves (took a loser)
fd = realized[~appr & (realized > 0)]         # false declines (skipped a winner)
print("=" * 80)
print(f"VAL P&L decomposition (n={len(V)}, ground-truthed)")
print(f"  oracle (perfect)          ${oracle/1e3:8,.0f}K")
print(f"  ours (E[NPV]>0)           ${ours/1e3:8,.0f}K   approve {appr.mean():.3f}")
print(f"  approve-all               ${appr_all/1e3:8,.0f}K")
print(f"  gap to oracle             ${(oracle-ours)/1e3:8,.0f}K")
print(f"   -- false APPROVES  n={len(fa):4d}  cost ${fa.sum()/1e3:8,.0f}K  (loosen=worse)")
print(f"   -- false DECLINES  n={len(fd):4d}  miss ${fd.sum()/1e3:8,.0f}K  (recoverable if rankable)")
# are false-declines profitable late-defaulters or just good loans?
fd_mask = ~appr & (realized > 0)
fd_def = fd_mask & dfl
print(f"      of false-declines: {fd_def.sum()} are profitable DEFAULTERS "
      f"(late, mean t*={np.nanmean(tstar[fd_def]):.0f}d), {fd_mask.sum()-fd_def.sum()} are repayers")

# ---------- 2. tau sweep: is the THRESHOLD the bottleneck? ----------
best = max(((realized[enpv > t].sum(), t) for t in np.linspace(-2000, 4000, 121)))
print(f"\n  best-tau P&L (val-optimal buffer)  ${best[0]/1e3:8,.0f}K  at tau=${best[1]:,.0f}"
      f"   -> threshold headroom ${(best[0]-ours)/1e3:,.0f}K")
# perfect-PD ranking, our timing: rank by realized, take all positives we can identify by PD order
order = np.argsort(cp)                          # approve safest-first
cum = np.cumsum(realized[order])
print(f"  best PD-ranked cutoff P&L          ${cum.max()/1e3:8,.0f}K  "
      f"-> ranking headroom vs ours ${(cum.max()-ours)/1e3:,.0f}K")

# ---------- 3. within-window drift (Deliverable B lever) ----------
print("\n" + "=" * 80)
print("DRIFT: default rate by application month (label shift continues?)")
for nm, df in [("train", L), ("val", V)]:
    t = pd.to_datetime(df.application_timestamp)
    g = df.assign(m=t.dt.to_period("M")).groupby("m").default_flag.agg(["mean", "size"])
    print(f" {nm}:", " ".join(f"{str(k)[2:]}:{r['mean']:.3f}" for k, r in g.iterrows()))
# val by cohort week within the test-like window
t = pd.to_datetime(V.application_timestamp)
wk = ((t - t.min()).dt.days // 7).clip(0, 12)
g = V.assign(w=wk).groupby("w").default_flag.agg(["mean", "size"])
sl = np.polyfit(g.index, g["mean"], 1)[0]
print(f" val cohort-week default rate:", " ".join(f"{int(k)}:{r['mean']:.2f}" for k, r in g.iterrows()))
print(f"  -> slope {sl:+.4f}/week ({sl*13:+.3f} over a 13-week window)  "
      f"current B uses a flat per-cohort level from model PD")
