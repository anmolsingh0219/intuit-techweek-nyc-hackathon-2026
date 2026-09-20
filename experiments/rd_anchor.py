"""Regression-discontinuity anchor: the prior lender's cutoff is a free natural
experiment. Loans JUST ABOVE the score cutoff ~ the just-declined ones except for
funding, so their realized default rate anchors what the declined zone would do —
an assumption-light check on our extrapolation (no IPW needed)."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src import data, survival

tr = data.load_train()
appr = data.approval_indicator(tr).values            # 1 = prior-approved
score = pd.to_numeric(tr["prior_underwriter_score"], errors="coerce").values
lab = tr["default_flag"].notna().values
y = tr["default_flag"].fillna(0).astype(int).values

# 1) recover the cutoff (near-deterministic threshold)
s_app, s_dec = score[appr == 1], score[appr == 0]
cut = (s_app.min() + s_dec.max()) / 2
overlap = ((s_dec > s_app.min()).mean() + (s_app < s_dec.max()).mean())
print("=" * 92)
print(f"PRIOR CUTOFF ~ {cut:.4f}  | approved score [{s_app.min():.3f},{s_app.max():.3f}] "
      f"declined [{s_dec.min():.3f},{s_dec.max():.3f}]")
print(f"overlap mass (declined-above-min-approved + approved-below-max-declined) = {overlap:.4f}  (~0 => no positivity)")

# 2) RD anchor: realized default rate of marginal approvals just above the cutoff
for band in (0.02, 0.05, 0.10):
    m = (appr == 1) & lab & (score <= cut + band)
    base = (appr == 1) & lab
    print(f"  marginal-approved [cut, cut+{band:.2f}]  n={m.sum():5d}  default={y[m].mean():.3f}"
          f"   (vs all-approved {y[base].mean():.3f})")

# 3) does our model's EXTRAPOLATED PD for the just-declined match the anchor?
L = tr[tr.default_flag.notna()].copy()
model = survival.RiskModel("logistic", recovery_rate=survival.recovery_rate_from(tr)).fit(L)
band = 0.05
anchor = y[(appr == 1) & lab & (score <= cut + band)].mean()       # realized, marginal approvals
just_dec = tr[(appr == 0) & (score >= cut - band)]                 # marginal declines (no label)
just_app = tr[(appr == 1) & (score <= cut + band)]
pd_dec, _ = model.predict_pd(just_dec)
pd_app, _ = model.predict_pd(just_app)
print("\nRD VALIDATION of declined-zone extrapolation:")
print(f"  realized default of marginal APPROVED (anchor)   {anchor:.3f}")
print(f"  our model PD on marginal APPROVED (should ~match) {pd_app.mean():.3f}")
print(f"  our model PD on marginal DECLINED (extrapolated)  {pd_dec.mean():.3f}  n={len(just_dec)}")
print(f"  deeper-declined PD (score<cut-0.10)               "
      f"{model.predict_pd(tr[(appr==0)&(score<cut-0.10)])[0].mean():.3f}")
print("\nRead: marginal-declined PD should sit at/above the anchor and rise as score falls.")
