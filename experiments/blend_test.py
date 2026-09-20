"""Does blending the top models beat the best single? (logistic / hgb / WOE-scorecard)"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import KFold
from src import data, features, npv, survival, woe
from src.models import make_backbone, proba

tr, va = data.load_train(), data.load_validation()
L = tr[tr.default_flag.notna()].copy(); V = va[va.default_flag.notna()].copy()
y, yv = L.default_flag.astype(int).values, V.default_flag.astype(int).values
rec = survival.recovery_rate_from(tr)
d = L[L.default_flag == 1]; wk = np.ceil(d.days_to_default.values/7).clip(1,13).astype(int)
marg = np.bincount(wk, minlength=14)[1:14].astype(float); marg /= marg.sum()
day_pw = np.array([d.days_to_default.values[wk==k].mean() if (wk==k).any() else 7*k-3 for k in range(1,14)])
R = V.requested_amount.values.astype(float); dfl = (V.default_flag==1).values; tdef = V.days_to_default.values

def frame(df): return pd.concat([df, features._engineer(df)], axis=1)
Xtr, fs = features.build_features(L); Xva,_ = features.build_features(V, fs)
enc = woe.WOEEncoder().fit(frame(L), y); Wtr, Wva = enc.transform(frame(L)), enc.transform(frame(V))

preds = {}
m = make_backbone("logistic", fs.columns); m.fit(Xtr, y); preds["log"] = proba(m, Xva)
hm = make_backbone("hgb", fs.columns); hm.fit(Xtr, y); preds["hgb"] = proba(hm, Xva)
lr = LogisticRegression(C=1.0, max_iter=2000).fit(Wtr.values, y); preds["woe"] = lr.predict_proba(Wva.values)[:,1]

def logit(p): p=np.clip(p,1e-4,1-1e-4); return np.log(p/(1-p))
def score(pv):
    cp=np.zeros(len(pv))
    for a,b in KFold(2,shuffle=True,random_state=0).split(pv):
        c=LogisticRegression().fit(logit(pv[a]).reshape(-1,1),yv[a]); cp[b]=c.predict_proba(logit(pv[b]).reshape(-1,1))[:,1]
    pi=cp[:,None]*marg[None,:]; enpv=npv.expected_npv_weekly(R,pi,day_pw,rec); appr=enpv>0
    realized=npv.realized_npv(R,dfl,tdef,rec)
    return roc_auc_score(yv,pv), brier_score_loss(yv,pv), appr.mean(), realized[appr].sum()

cands = {"log":preds["log"], "hgb":preds["hgb"], "woe":preds["woe"],
         "log+hgb":(preds["log"]+preds["hgb"])/2,
         "log+hgb+woe":(preds["log"]+preds["hgb"]+preds["woe"])/3,
         "logit-avg(log+hgb)":1/(1+np.exp(-(logit(preds["log"])+logit(preds["hgb"]))/2)),
         "0.5log+0.5hgb+woe*0":(preds["log"]+preds["hgb"])/2}
rows=[]
for k,pv in cands.items():
    a,br,ar,pnl=score(pv); rows.append(dict(model=k,AUC=round(a,4),Brier=round(br,4),approve=round(ar,3),val_PnL=f"${pnl/1e3:,.0f}K"))
print(pd.DataFrame(rows).to_string(index=False))
print(f"\npairwise corr  log~hgb {np.corrcoef(preds['log'],preds['hgb'])[0,1]:.3f} | "
      f"log~woe {np.corrcoef(preds['log'],preds['woe'])[0,1]:.3f} | hgb~woe {np.corrcoef(preds['hgb'],preds['woe'])[0,1]:.3f}")
