"""Column profiling + model bake-off: WOE-scorecard logistic vs current logistic
vs trees vs MLP (DL). Fair val comparison: AUC / Brier / calibrated val P&L on a
shared marginal-timing NPV decision. Also answers 'is DL worth it'.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

from src import data, features, npv, survival, woe  # noqa: E402
from src.models import make_backbone, proba  # noqa: E402

pd.set_option("display.width", 200, "display.max_columns", 40)
tr, va = data.load_train(), data.load_validation()
dd = data.data_dictionary().set_index("field")
L = tr[tr.default_flag.notna()].copy()
V = va[va.default_flag.notna()].copy()
y, yv = L.default_flag.astype(int).values, V.default_flag.astype(int).values
rec = survival.recovery_rate_from(tr)

# shared timing (marginal week dist + representative day) from train defaulters
d = L[L.default_flag == 1]
wk = np.ceil(d.days_to_default.values / 7.0).clip(1, 13).astype(int)
marg = np.bincount(wk, minlength=14)[1:14].astype(float)
marg /= marg.sum()
day_pw = np.array([d.days_to_default.values[wk == k].mean() if (wk == k).any() else 7 * k - 3
                   for k in range(1, 14)])

# ======================= 1. COLUMN PROFILE =======================
print("=" * 100)
print(f"COLUMN PROFILE  | labeled train {len(L):,} (def {y.mean():.3f}) | "
      f"val {len(V):,} (def {yv.mean():.3f}) | recovery {rec:.4f}")
print("=" * 100)
prof = []
for c in data.feature_columns():
    if c == "application_timestamp":
        continue
    s = L[c]
    prof.append(dict(field=c, group=dd.loc[c, "group"], dtype=dd.loc[c, "dtype"],
                     interv=str(dd.loc[c, "intervenable"])[0], null=f"{s.isna().mean()*100:.0f}%",
                     nuniq=int(s.nunique()), meaning=str(dd.loc[c, "notes"])[:58]))
print(pd.DataFrame(prof).to_string(index=False))

print("\n--- categorical level -> default rate (n) ---")
for c in ["sector", "geography_region", "intended_use_of_funds", "application_channel",
          "owner_personal_credit_band", "employee_count_bucket", "has_linked_bank_feed"]:
    g = L.groupby(c).default_flag.agg(["mean", "size"])
    parts = " | ".join(f"{k}:{r['mean']:.3f}(n={int(r['size'])})" for k, r in g.iterrows())
    print(f"{c:32s} {parts}")

# ======================= 2. FEATURE MATRICES =======================
def frame(df):
    return pd.concat([df, features._engineer(df)], axis=1)

Xtr, fs = features.build_features(L)
Xva, _ = features.build_features(V, fs)
enc = woe.WOEEncoder().fit(frame(L), y)
Wtr, Wva = enc.transform(frame(L)), enc.transform(frame(V))
print("\n--- WOE scorecard feature set (IV) ---  cols:", Wtr.shape[1])
print(enc.iv_table().T.to_string())

# ======================= 3. BAKE-OFF =======================
def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))

def cal_pnl(pv):
    """2-fold Platt-cross-fit calibration -> E[NPV]>0 decision -> realized val P&L."""
    cp = np.zeros(len(pv))
    for a, b in KFold(2, shuffle=True, random_state=0).split(pv):
        lr = LogisticRegression().fit(logit(pv[a]).reshape(-1, 1), yv[a])
        cp[b] = lr.predict_proba(logit(pv[b]).reshape(-1, 1))[:, 1]
    pi = cp[:, None] * marg[None, :]
    enpv = npv.expected_npv_weekly(V.requested_amount.values.astype(float), pi, day_pw, rec)
    appr = enpv > 0
    realized = npv.realized_npv(V.requested_amount.values.astype(float),
                                (V.default_flag == 1).values, V.days_to_default.values, rec)
    return appr.mean(), realized[appr].sum(), cp.mean() - yv.mean()

def evaluate(name, pv):
    auc = roc_auc_score(yv, pv)
    brier = brier_score_loss(yv, pv)
    ar, pnl, cg = cal_pnl(pv)
    return dict(model=name, AUC=round(auc, 4), Brier=round(brier, 4),
                cal_gap=round(cg, 4), approve=round(ar, 3), val_PnL=f"${pnl/1e3:,.0f}K")

rows = []
# current production-style logistic (bagged 5) on full feature set
m = make_backbone("logistic", fs.columns)
m.fit(Xtr, y)
rows.append(evaluate("current_logistic(73f)", proba(m, Xva)))
# WOE scorecard logistic
lr = LogisticRegression(C=1.0, max_iter=2000).fit(Wtr.values, y)
pv_woe = lr.predict_proba(Wva.values)[:, 1]
rows.append(evaluate("WOE_logistic(19f)", pv_woe))
# trees (reference)
for nm in ["xgb", "hgb"]:
    try:
        tm = make_backbone(nm, fs.columns)
        tm.fit(Xtr, y)
        rows.append(evaluate(nm + "(73f)", proba(tm, Xva)))
    except Exception as e:
        print("skip", nm, e)
try:
    import lightgbm  # noqa
    lm = make_backbone("lgbm", fs.columns); lm.fit(Xtr, y)
    rows.append(evaluate("lgbm(73f)", proba(lm, Xva)))
except Exception as e:
    print("skip lgbm:", e)
# ===== DL: MLPs =====
mlp_w = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=300,
                      early_stopping=True, random_state=0).fit(Wtr.values, y)
rows.append(evaluate("MLP_on_WOE(19f)", mlp_w.predict_proba(Wva.values)[:, 1]))
# MLP on raw imputed+scaled (full)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
imp = SimpleImputer(strategy="median").fit(Xtr)
sc = StandardScaler().fit(imp.transform(Xtr))
Zt, Zv = sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xva))
mlp_r = MLPClassifier(hidden_layer_sizes=(128, 64, 32), alpha=1e-3, max_iter=300,
                      early_stopping=True, random_state=0).fit(Zt, y)
rows.append(evaluate("MLP_on_raw(73f)", mlp_r.predict_proba(Zv)[:, 1]))

print("\n" + "=" * 100)
print("BAKE-OFF (val n=2,551; calibrated 2-fold Platt; E[NPV]>0 on shared marginal timing)")
print("=" * 100)
print(pd.DataFrame(rows).to_string(index=False))

# ======================= 4. SCORECARD COEFFICIENTS =======================
print("\n--- WOE-logistic scorecard (signed coef; +coef on +WOE => safer feature lowers risk) ---")
coef = pd.Series(lr.coef_[0], index=[c[:-4] for c in Wtr.columns]).sort_values(key=abs, ascending=False)
print(coef.round(3).to_string())
print(f"intercept {lr.intercept_[0]:.3f}")
