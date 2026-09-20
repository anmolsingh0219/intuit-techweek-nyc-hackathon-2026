"""Two-mode default structure: defaults are either EARLY (missed ACH draws --
catastrophic) or at the DAY-90 balance check (most draws already collected --
near-profitable). The real underwriting question is P(early default), not P(default)."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src import data, features, npv, survival

tr = data.load_train(); L = tr[tr.default_flag.notna()].copy()
rec = survival.recovery_rate_from(tr)
d = L[L.default_flag == 1]
t = d.days_to_default.values
R = d.requested_amount.values.astype(float)

print("=" * 92)
print("DEFAULT-TIMING HISTOGRAM (days_to_default among defaulters):")
h, edges = np.histogram(t, bins=[1,7,14,21,28,35,42,49,56,63,70,77,84,91])
for i in range(len(h)):
    print(f"  day {int(edges[i]):3d}-{int(edges[i+1]):3d}: {h[i]:5d}  {'#'*(h[i]//40)}")

realized = npv.realized_npv(R, np.ones(len(d), bool), t, rec)
early = t <= 60; late = t > 60
print("\nMEAN realized NPV per $ principal:")
print(f"  EARLY default (t*<=60, missed draws)  n={early.sum():5d}  {(realized[early]/R[early]).mean():+.3f}  (catastrophic)")
print(f"  DAY-90 default (t*>60, balance check)  n={late.sum():5d}  {(realized[late]/R[late]).mean():+.3f}  (~ repaid)")
print(f"  share of defaults that are DAY-90: {late.mean():.3f}  -> these barely cost us")
print(f"  TRUE costly-default rate (early only) = {early.sum()/len(L):.3f}  vs headline default {len(d)/len(L):.3f}")

y_def = L.default_flag.astype(int).values
y_early = ((L.default_flag == 1) & (L.days_to_default <= 60)).astype(int).values
X, fs = features.build_features(L)
def cv_auc(yv):
    p = np.zeros(len(yv))
    for a, b in KFold(5, shuffle=True, random_state=0).split(X):
        mdl = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                        ("m", LogisticRegression(C=0.5, max_iter=2000))])
        mdl.fit(X.iloc[a], yv[a]); p[b] = mdl.predict_proba(X.iloc[b])[:, 1]
    return roc_auc_score(yv, p)
print(f"\n5-fold AUC  P(any default)   = {cv_auc(y_def):.4f}")
print(f"5-fold AUC  P(EARLY default) = {cv_auc(y_early):.4f}   <- the decision-relevant target")
print(f"base rates: any-default {y_def.mean():.3f}  early-default {y_early.mean():.3f}")
print("\nRead: if P(early) is more separable, screening on the EARLY mode sharpens the decision.")
