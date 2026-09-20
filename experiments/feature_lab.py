"""experiments/feature_lab.py - Credit-risk feature exploration (WOE / IV scorecard view).

For every raw + engineered feature, on the LABELED training book (prior-approved &
matured, the only rows with default_flag), this computes the classic scorecard view:

  * meaning (from data_dictionary), group, type, null %, cardinality, basic stats
  * supervised MONOTONIC binning  (quantile -> pool-adjacent-violators merge)
  * Weight of Evidence (WOE) + Information Value (IV)  = the credit-risk "weight"
  * univariate AUC cross-check
  * train-vs-val IV stability + PSI                    (ties to the temporal-drift finding)
  * encoding / binning recommendation
  * MNAR missingness default-rate split
  * redundancy (Spearman) clustering of the strongest features

Outputs: console summary + reports/feature_analysis.md + reports/feature_iv_table.csv

WOE convention here:  WOE = ln(%non-default / %default).  POSITIVE WOE = SAFER bin.
IV interpretation (Siddiqi): <0.02 useless | 0.02-0.1 weak | 0.1-0.3 medium |
0.3-0.5 strong | >0.5 suspiciously strong (often leakage / selection).
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
from sklearn.metrics import roc_auc_score  # noqa: E402

from src import data  # noqa: E402
from src.features import _engineer  # noqa: E402

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

ORDINAL = {"employee_count_bucket", "owner_personal_credit_band"}
NOMINAL = {"sector", "geography_region", "intended_use_of_funds", "application_channel"}
BINARY = {"has_linked_bank_feed"}
# selection / leakage-ish: analyze but flag (do NOT put in the model)
SELECTION = {"prior_underwriter_score", "prior_approved_amount"}
SKIP = {"application_timestamp", "prior_decision", "business_id", "applicant_id"}


def iv_label(iv: float) -> str:
    for thr, lab in [(0.02, "useless"), (0.1, "weak"), (0.3, "medium"), (0.5, "strong")]:
        if iv < thr:
            return lab
    return "suspicious"


# --------------------------------------------------------------------------- #
# WOE / IV core
# --------------------------------------------------------------------------- #
def woe_iv(bins, y, eps: float = 0.5):
    d = pd.DataFrame({"b": np.asarray(bins), "y": np.asarray(y)})
    g = d.groupby("b", observed=True)["y"]
    t = pd.DataFrame({"n": g.size(), "bad": g.sum()})
    t["good"] = t["n"] - t["bad"]
    tb, tg, k = t["bad"].sum(), t["good"].sum(), len(t)
    t["bad_rate"] = t["bad"] / t["n"]
    t["dist_bad"] = (t["bad"] + eps) / (tb + eps * k)
    t["dist_good"] = (t["good"] + eps) / (tg + eps * k)
    t["woe"] = np.log(t["dist_good"] / t["dist_bad"])
    t["iv"] = (t["dist_good"] - t["dist_bad"]) * t["woe"]
    return t, float(t["iv"].sum())


def monotone_edges(x, y, q: int = 10, min_frac: float = 0.05):
    """Quantile bins, then pool-adjacent-violators to a monotonic bad-rate trend."""
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
        df = pd.DataFrame({"i": idx, "y": yv}).groupby("i")["y"]
        return df.size(), df.mean()

    # 1) enforce minimum bin mass
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

    # 2) enforce monotonic bad-rate
    sz, rt = rates(edges)
    r = np.array([rt.get(i, np.nan) for i in range(len(edges) - 1)])
    r = r[np.isfinite(r)]
    if len(r) > 1 and np.ptp(r) > 0:
        direction = np.sign(np.corrcoef(np.arange(len(r)), r)[0, 1])
    else:
        direction = 1.0
    if direction == 0:
        direction = 1.0
    guard = 0
    while len(edges) > 3 and guard < 60:
        guard += 1
        sz, rt = rates(edges)
        r = np.array([rt.get(i, np.nan) for i in range(len(edges) - 1)])
        diffs = np.diff(r) * direction
        bad = np.where(diffs < 0)[0]
        if len(bad) == 0:
            break
        edges = np.delete(edges, bad[0] + 1)
    return edges


def numeric_labels(series, edges):
    x = series.values.astype(float)
    lab = np.full(len(x), "MISSING", dtype=object)
    m = np.isfinite(x)
    idx = np.digitize(x[m], edges[1:-1])
    names = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        lo_s = "-inf" if lo == -np.inf else f"{lo:.4g}"
        hi_s = "inf" if hi == np.inf else f"{hi:.4g}"
        names.append(f"{i}:[{lo_s},{hi_s})")
    lab[m] = [names[i] for i in idx]
    return pd.Series(lab, index=series.index)


def cat_labels(series, min_frac: float = 0.01):
    s = series.astype("object").where(series.notna(), "MISSING")
    vc = s.value_counts(dropna=False)
    rare = vc[vc < max(min_frac * len(s), 20)].index
    return s.where(~s.isin(rare), "OTHER")


def uni_auc(x, y):
    x = np.asarray(x, float)
    if not np.isfinite(x).any():
        return np.nan
    xf = np.where(np.isfinite(x), x, np.nanmedian(x[np.isfinite(x)]))
    try:
        a = roc_auc_score(y, xf)
    except Exception:
        return np.nan
    return max(a, 1 - a)


def psi(exp_counts, act_counts, eps=1e-6):
    e = np.asarray(exp_counts, float)
    a = np.asarray(act_counts, float)
    e = np.clip(e / e.sum(), eps, None)
    a = np.clip(a / a.sum(), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


# --------------------------------------------------------------------------- #
# Load labeled books
# --------------------------------------------------------------------------- #
tr = data.load_train()
va = data.load_validation()
dd = data.data_dictionary().set_index("field")


def labeled(df):
    return df[df["default_flag"].notna()].copy()


L, Lv = labeled(tr), labeled(va)
y = L["default_flag"].astype(int).to_numpy()
yv = Lv["default_flag"].astype(int).to_numpy()
base_rate, base_rate_v = y.mean(), yv.mean()

# engineered features
eng = _engineer(L)
eng_v = _engineer(Lv)

feat_cols = [c for c in data.feature_columns() if c not in SKIP]


def kind(col):
    if col in NOMINAL:
        return "nominal"
    if col in ORDINAL:
        return "ordinal"
    if col in BINARY:
        return "binary"
    return "numeric"


rows = []
woe_tables = {}  # col -> (train table, type)
for col in feat_cols:
    s, sv = L[col], Lv[col]
    k = kind(col)
    null_pct = float(s.isna().mean() * 100)
    nun = int(s.nunique(dropna=True))
    try:
        if k in ("nominal", "ordinal", "binary"):
            lab = cat_labels(s)
            t, iv = woe_iv(lab, y)
            # val IV with train levels
            lab_v = sv.astype("object").where(sv.notna(), "MISSING")
            lab_v = lab_v.where(lab_v.isin(t.index), "OTHER")
            _, iv_v = woe_iv(lab_v, yv) if lab_v.nunique() > 1 else (None, np.nan)
            # ordinal monotonicity vs code order
            mono = ""
            if k == "ordinal":
                br = L.assign(_b=pd.to_numeric(s, errors="coerce")).groupby("_b")["default_flag"].mean()
                if br.notna().sum() > 2:
                    sp = pd.Series(br.index).corr(pd.Series(br.values), method="spearman")
                    mono = f"rank_corr={sp:+.2f}"
            # PSI
            tr_counts = lab.value_counts()
            va_counts = lab_v.value_counts().reindex(tr_counts.index).fillna(0)
            ps = psi(tr_counts.values, va_counts.values)
            auc = np.nan
        else:
            edges = monotone_edges(s.values, y)
            if edges is None:
                lab = cat_labels(s.astype("object"))
                t, iv = woe_iv(lab, y)
                iv_v, ps, mono = np.nan, np.nan, "n/a"
            else:
                lab = numeric_labels(s, edges)
                t, iv = woe_iv(lab, y)
                lab_v = numeric_labels(sv, edges)
                _, iv_v = woe_iv(lab_v, yv)
                # monotonic? (exclude MISSING)
                tt = t[t.index != "MISSING"].copy()
                tt = tt.reindex(sorted(tt.index, key=lambda z: int(z.split(":")[0])))
                mono = "monotone" if (np.all(np.diff(tt["woe"]) >= -1e-9) or
                                      np.all(np.diff(tt["woe"]) <= 1e-9)) else "non-monotone"
                trc = lab.value_counts()
                vac = lab_v.value_counts().reindex(trc.index).fillna(0)
                ps = psi(trc.values, vac.values)
            auc = uni_auc(s.values, y)
        woe_tables[col] = (t, k)
        rows.append(dict(feature=col, group=dd.loc[col, "group"], type=k,
                         intervenable=str(dd.loc[col, "intervenable"]), null_pct=round(null_pct, 1),
                         n_unique=nun, IV=round(iv, 4), IV_band=iv_label(iv),
                         IV_val=round(iv_v, 4) if np.isfinite(iv_v) else np.nan,
                         PSI=round(ps, 3) if np.isfinite(ps) else np.nan,
                         uni_AUC=round(auc, 4) if np.isfinite(auc) else np.nan, note=mono))
    except Exception as e:  # keep going
        rows.append(dict(feature=col, group=dd.loc[col, "group"], type=k, IV=np.nan, note=f"ERR {e}"))

# engineered
for col in eng.columns:
    s, sv = eng[col], eng_v[col] if col in eng_v else pd.Series(np.nan, index=Lv.index)
    try:
        edges = monotone_edges(s.values, y)
        if edges is None:
            continue
        lab = numeric_labels(s, edges)
        t, iv = woe_iv(lab, y)
        lab_v = numeric_labels(sv, edges)
        _, iv_v = woe_iv(lab_v, yv)
        ps = psi(lab.value_counts().values,
                 lab_v.value_counts().reindex(lab.value_counts().index).fillna(0).values)
        auc = uni_auc(s.values, y)
        woe_tables["ENG:" + col] = (t, "engineered")
        rows.append(dict(feature="ENG:" + col, group="engineered", type="numeric",
                         intervenable="-", null_pct=round(float(s.isna().mean() * 100), 1),
                         n_unique=int(s.nunique()), IV=round(iv, 4), IV_band=iv_label(iv),
                         IV_val=round(iv_v, 4) if np.isfinite(iv_v) else np.nan,
                         PSI=round(ps, 3) if np.isfinite(ps) else np.nan,
                         uni_AUC=round(auc, 4) if np.isfinite(auc) else np.nan, note=""))
    except Exception as e:
        rows.append(dict(feature="ENG:" + col, group="engineered", IV=np.nan, note=f"ERR {e}"))

R = pd.DataFrame(rows).sort_values("IV", ascending=False, na_position="last").reset_index(drop=True)
R.to_csv(REPORTS / "feature_iv_table.csv", index=False)

# --------------------------------------------------------------------------- #
# MNAR: default rate present vs missing for structurally-nullable columns
# --------------------------------------------------------------------------- #
mnar = []
for col in data.nullable_feature_columns():
    if col not in L:
        continue
    miss = L[col].isna()
    if miss.sum() == 0 or (~miss).sum() == 0:
        continue
    mnar.append(dict(feature=col, pct_missing=round(miss.mean() * 100, 1),
                     dr_present=round(y[~miss.values].mean(), 4),
                     dr_missing=round(y[miss.values].mean(), 4),
                     gap=round(y[miss.values].mean() - y[~miss.values].mean(), 4)))
MNAR = pd.DataFrame(mnar).sort_values("gap", key=lambda s: s.abs(), ascending=False)

# --------------------------------------------------------------------------- #
# Redundancy among the strongest numeric/engineered features (Spearman > .7)
# --------------------------------------------------------------------------- #
top_num = [f for f in R["feature"].head(25)
           if (f.startswith("ENG:") or kind(f) == "numeric")]
mat = {}
for f in top_num:
    mat[f] = (eng[f[4:]] if f.startswith("ENG:") else L[f]).astype(float).values
C = pd.DataFrame(mat).corr(method="spearman")
pairs = []
for i in range(len(C)):
    for j in range(i + 1, len(C)):
        if abs(C.iloc[i, j]) >= 0.7:
            pairs.append((C.index[i], C.columns[j], round(C.iloc[i, j], 2)))

# --------------------------------------------------------------------------- #
# Console summary
# --------------------------------------------------------------------------- #
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 100)
print("=" * 92)
print(f"LABELED TRAINING BOOK: {len(L):,} rows | default rate {base_rate:.4f} | "
      f"val labeled {len(Lv):,} rows | val default rate {base_rate_v:.4f}")
print("=" * 92)
print("\nFEATURE 'WEIGHTS' = INFORMATION VALUE (ranked).  PSI>0.25 => distribution drift train->val\n")
show = R[["feature", "group", "type", "null_pct", "n_unique", "IV", "IV_band",
          "IV_val", "PSI", "uni_AUC", "note"]]
print(show.to_string(index=False))
print("\n" + "=" * 92)
print("MNAR missingness (default rate when the field is PRESENT vs MISSING):")
print(MNAR.to_string(index=False))
print("\n" + "=" * 92)
print("REDUNDANT pairs among strongest features (|Spearman| >= 0.70) -> keep one each:")
for a, b, c in pairs:
    print(f"   {c:+.2f}   {a}   <->   {b}")

# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def md_table(df):
    cols = list(df.columns)
    out = "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return out


lines = []
lines.append("# Feature Lab - WOE / IV scorecard view of every feature\n")
lines.append(f"Labeled training book (prior-approved & matured): **{len(L):,} rows**, "
             f"default rate **{base_rate:.3f}**. Validation labeled: **{len(Lv):,} rows**, "
             f"default rate **{base_rate_v:.3f}** (the upward drift).\n")
lines.append("WOE = ln(%non-default / %default); **positive WOE = safer bin**. "
             "IV is the feature's predictive *weight*: <0.02 useless, 0.02-0.1 weak, "
             "0.1-0.3 medium, 0.3-0.5 strong, >0.5 suspicious (leakage/selection).\n")
lines.append("## 1. Feature weights (Information Value), ranked\n")
lines.append(md_table(show))
lines.append("\n## 2. WOE / monotonic bin tables for the strongest features\n")
strongest = [f for f in R["feature"].head(14) if f in woe_tables]
for f in strongest:
    t, kk = woe_tables[f]
    disp = t.copy()
    if kk != "engineered" and kk != "numeric":
        disp = disp.sort_values("bad_rate")
    else:
        def _key(z):
            try:
                return int(str(z).split(":")[0])
            except Exception:
                return 999
        disp = disp.reindex(sorted(disp.index, key=_key))
    disp = disp.reset_index().rename(columns={"index": "bin", "b": "bin"})
    disp = disp[["bin", "n", "bad", "bad_rate", "woe", "iv"]].round(
        {"bad_rate": 4, "woe": 3, "iv": 4})
    meaning = str(dd.loc[f[4:] if f.startswith("ENG:") else f, "notes"])[:140] if (
        (f[4:] if f.startswith("ENG:") else f) in dd.index) else "engineered ratio"
    lines.append(f"\n### {f}  (IV={R.loc[R.feature==f,'IV'].values[0]})\n")
    lines.append(f"_{meaning}_\n\n")
    lines.append(md_table(disp))
lines.append("\n## 3. MNAR missingness (signal in the *fact* of missing)\n")
lines.append(md_table(MNAR))
lines.append("\n## 4. Redundancy (|Spearman| >= 0.70) - keep one per cluster\n")
lines.append("\n".join(f"- `{a}` <-> `{b}`  ({c:+.2f})" for a, b, c in pairs) or "- none")
(REPORTS / "feature_analysis.md").write_text("\n".join(lines), encoding="utf-8")
print("\nWrote reports/feature_analysis.md  and  reports/feature_iv_table.csv")
