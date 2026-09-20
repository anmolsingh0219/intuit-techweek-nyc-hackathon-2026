"""Column-connection map: cluster features by similarity (Spearman) to reveal the
latent factors, the redundancy, and the one-representative-per-cluster lean set."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from src import data, features

tr = data.load_train(); L = tr[tr.default_flag.notna()].copy()
y = L.default_flag.astype(int).values
dd = data.data_dictionary().set_index("field")
frame = pd.concat([L, features._engineer(L)], axis=1)

# IV lookup (from feature_lab)
ivt = pd.read_csv(ROOT / "reports" / "feature_iv_table.csv")
iv = {}
for _, r in ivt.iterrows():
    iv[str(r["feature"]).replace("ENG:", "")] = r["IV"]

# numeric/continuous features worth connecting (skip ids, outcomes, constants)
skip = set(data.OUTCOME_COLS) | set(data.ID_COLS) | {"application_timestamp", "prior_decision"}
num = [c for c in frame.columns if c not in skip and pd.api.types.is_numeric_dtype(frame[c])
       and frame[c].nunique() > 3]
X = frame[num].apply(pd.to_numeric, errors="coerce")
corr = X.corr("spearman").fillna(0)

# hierarchical clustering on 1-|rho|
dist = 1 - corr.abs()
np.fill_diagonal(dist.values, 0)
Z = linkage(squareform(dist.values, checks=False), method="average")
cl = fcluster(Z, t=0.35, criterion="distance")   # group features with avg |rho| > ~0.65

# target correlation (signed) for direction
tcorr = {c: np.corrcoef(X[c].fillna(X[c].median()), y)[0, 1] for c in num}

groups = {}
for c, g in zip(num, cl):
    groups.setdefault(g, []).append(c)

print("=" * 100)
print("COLUMN-CONNECTION MAP  — features grouped by |Spearman| > ~0.65 (latent factors)")
print("=" * 100)
ranked = sorted(groups.values(), key=lambda m: -max(abs(iv.get(c, 0)) for c in m))
for i, mem in enumerate(ranked, 1):
    mem = sorted(mem, key=lambda c: -abs(iv.get(c, 0)))
    rep = mem[0]
    dom = dd.loc[rep, "group"] if rep in dd.index else "engineered"
    head = f"[F{i}] repr={rep}  IV={iv.get(rep,0):.3f}  domain={dom}  (corr w/default {tcorr[rep]:+.2f})"
    print("\n" + head)
    if len(mem) > 1:
        for c in mem[1:]:
            print(f"      ~ {c:38s} IV={iv.get(c,0):.3f}  rho_target={tcorr[c]:+.2f}")
    else:
        print("      (standalone — no near-duplicate)")

print("\n" + "=" * 100)
print("LEAN SET = one representative per factor (drop the rest as redundant):")
print("  " + ", ".join(sorted([sorted(m, key=lambda c:-abs(iv.get(c,0)))[0] for m in ranked],
                               key=lambda c: -abs(iv.get(c, 0)))))
