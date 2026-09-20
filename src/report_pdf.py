"""Generate reports/data_and_logic_overview.pdf.

A self-contained explainer: what the data is, the five things we must account
for, the loan economics (why default *timing* drives everything), the empirical
default trajectory, and the end-to-end pipeline logic that ties A/B/C/D together.

Figures are rendered from the real data with matplotlib; the document is laid
out with reportlab (no LaTeX needed).  Run:  python -m src.report_pdf
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

from . import data, npv, paths

FIG_DIR = paths.REPORTS_DIR / "figures"
PDF_PATH = paths.REPORTS_DIR / "data_and_logic_overview.pdf"

BLUE = colors.HexColor("#0a6ed1")
DARK = colors.HexColor("#1a1a2e")
GREY = colors.HexColor("#5a5a6a")
LIGHT = colors.HexColor("#eef2f7")
RED = colors.HexColor("#c0392b")
GREEN = colors.HexColor("#1e8449")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _save(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def fig_npv_curve() -> str:
    R = 10_000.0
    days = np.arange(1, 61)
    vals = npv.npv_default(R, days)
    repaid = npv.npv_repaid(R)
    be = days[np.argmin(np.abs(vals))]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.axhline(0, color="grey", lw=0.8)
    ax.plot(days, vals, color="#0a6ed1", lw=2.2, label="NPV if default on day t*")
    ax.axhline(repaid, color="#1e8449", ls="--", lw=1.5, label=f"NPV if repaid (+${repaid:,.0f})")
    ax.scatter([1, 30, 60], [npv.npv_default(R, d) for d in (1, 30, 60)], color="#c0392b", zorder=5)
    for d in (1, 30, 60):
        v = npv.npv_default(R, d)
        ax.annotate(f"${v:,.0f}", (d, v), textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.axvline(be, color="#c0392b", ls=":", lw=1, alpha=0.7)
    ax.set_title("Loan economics: profit vs. the day a $10K loan defaults", fontsize=11, weight="bold")
    ax.set_xlabel("Default day t*  (1 = immediate, 60 = end of term)")
    ax.set_ylabel("NPV (profit, $)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    return _save(fig, "npv_curve.png")


def fig_timing_hist(tr) -> str:
    dtd = tr["days_to_default"].dropna().astype(float)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.hist(dtd, bins=range(0, 95, 3), color="#0a6ed1", alpha=0.85, edgecolor="white")
    ax.axvline(60, color="#c0392b", ls="--", lw=1.3, label="term ends (day 60)")
    ax.axvline(dtd.median(), color="#1e8449", ls=":", lw=1.5, label=f"median = {dtd.median():.0f}d")
    ax.set_title("When do defaults happen?  days_to_default (approved loans)", fontsize=11, weight="bold")
    ax.set_xlabel("day of default"); ax.set_ylabel("count")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    return _save(fig, "timing_hist.png")


def fig_selection(tr) -> str:
    appr = data.approval_indicator(tr)
    s = tr["prior_underwriter_score"].astype(float)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.hist(s[appr == 1].dropna(), bins=40, alpha=0.6, color="#1e8449", label="approved (labeled)", density=True)
    ax.hist(s[appr == 0].dropna(), bins=40, alpha=0.6, color="#c0392b", label="declined (UNLABELED)", density=True)
    ax.set_title("Selective labels: prior underwriter score drives who got a label",
                 fontsize=11, weight="bold")
    ax.set_xlabel("prior_underwriter_score"); ax.set_ylabel("density")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    return _save(fig, "selection.png")


def fig_missingness(tr) -> str:
    nulls = tr[data.feature_columns()].isna().mean()
    nulls = nulls[nulls > 0].sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(range(len(nulls)), nulls.values * 100, color="#0a6ed1", alpha=0.85)
    ax.set_yticks(range(len(nulls)))
    ax.set_yticklabels(nulls.index, fontsize=7.5)
    ax.set_xlabel("null rate (%)")
    ax.set_title("Structural (MNAR) missingness by feature", fontsize=11, weight="bold")
    ax.grid(alpha=0.25, axis="x")
    return _save(fig, "missingness.png")


def fig_cumulative_default(tr) -> str:
    lab = data.build_survival_label(tr)
    m = lab["has_label"]
    n = int(m.sum())
    dtd = tr.loc[m & (lab["event"] == 1), "days_to_default"].astype(float)
    days = np.arange(1, 91)
    cdr = np.array([(dtd <= d).sum() / n for d in days])
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(days, cdr * 100, color="#0a6ed1", lw=2.2)
    ax.fill_between(days, 0, cdr * 100, color="#0a6ed1", alpha=0.12)
    for wk in (1, 4, 8, 13):
        d = 7 * wk
        ax.annotate(f"wk{wk}: {np.interp(d, days, cdr)*100:.1f}%", (d, np.interp(d, days, cdr) * 100),
                    textcoords="offset points", xytext=(3, 5), fontsize=7.5)
    ax.set_title("Empirical cumulative default rate vs. loan age (Deliverable B shape)",
                 fontsize=11, weight="bold")
    ax.set_xlabel("loan age (days)"); ax.set_ylabel("cumulative default %")
    ax.grid(alpha=0.25)
    return _save(fig, "cumulative_default.png")


def fig_pipeline() -> str:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, text, fc, tc="white", fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fc, ec="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
                fontsize=fs, weight="bold", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     lw=1.3, color="#5a5a6a"))

    box(0.2, 7.7, 2.3, 1.5, "Raw data\ntrain / val / test", "#1a1a2e")
    box(0.2, 5.6, 2.3, 1.4, "Feature pipeline\nMNAR flags, ratios,\nleakage drop", "#5a5a6a")
    box(0.2, 3.2, 2.3, 1.6, "Selection-bias\ncorrection (IPW)\npropensity of approval", "#8e44ad")
    box(3.4, 5.4, 2.6, 2.0, "SHARED SPINE\nDiscrete-time\nhazard model\nh(t), t=1..90", "#0a6ed1", fs=9)
    box(6.9, 7.7, 2.9, 1.5, "A — decisions\napprove iff E[NPV]>0\n+ predicted PD", "#1e8449")
    box(6.9, 5.6, 2.9, 1.5, "B — trajectory\ncohort cumulative\ndefault curve", "#1e8449")
    box(6.9, 3.5, 2.9, 1.5, "C — counterfactual\ndo(feature=v)\ncausal PD", "#c0392b")
    box(3.4, 2.0, 2.6, 1.2, "Calibration\nconformal 90% PIs\n(A, B, C)", "#d68910", fs=8)
    box(6.9, 1.4, 2.9, 1.3, "D — writeup\ndefends all choices\n(esp. causal)", "#1a1a2e")

    arrow(1.35, 7.7, 1.35, 7.0)
    arrow(1.35, 5.6, 1.35, 4.8)
    arrow(2.5, 6.3, 3.4, 6.4)           # features -> spine
    arrow(2.5, 4.0, 3.4, 5.6)           # IPW -> spine
    arrow(6.0, 6.7, 6.9, 8.0)           # spine -> A
    arrow(6.0, 6.4, 6.9, 6.3)           # spine -> B
    arrow(6.0, 5.6, 6.9, 4.2)           # spine -> C (features path)
    arrow(4.7, 5.4, 4.7, 3.2)           # spine -> calibration
    arrow(6.0, 2.6, 6.9, 2.1)           # calibration -> D
    ax.text(5.0, 9.4, "End-to-end logic", ha="center", fontsize=12, weight="bold", color="#1a1a2e")
    return _save(fig, "pipeline.png")


# --------------------------------------------------------------------------- #
# PDF assembly
# --------------------------------------------------------------------------- #

def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("H1b", parent=ss["Heading1"], textColor=BLUE, spaceBefore=14, spaceAfter=6, fontSize=15))
    ss.add(ParagraphStyle("H2b", parent=ss["Heading2"], textColor=DARK, spaceBefore=8, spaceAfter=4, fontSize=12))
    ss.add(ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.5, leading=13, spaceAfter=5, textColor=DARK))
    ss.add(ParagraphStyle("Small", parent=ss["BodyText"], fontSize=8, leading=10, textColor=GREY))
    ss.add(ParagraphStyle("BulletX", parent=ss["BodyText"], fontSize=9.5, leading=13, leftIndent=8))
    ss.add(ParagraphStyle("TitleBig", parent=ss["Title"], textColor=DARK, fontSize=24, leading=28))
    ss.add(ParagraphStyle("Sub", parent=ss["Normal"], textColor=BLUE, fontSize=12, alignment=TA_CENTER))
    return ss


def _img(path, width=6.6 * inch):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(path, width=width, height=width * h / w)


def _kv_table(rows, col_widths, header=True):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, LIGHT]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style += [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                  ("TEXTCOLOR", (0, 0), (-1, 0), BLUE)]
    t.setStyle(TableStyle(style))
    return t


def build_pdf() -> str:
    tr = data.load_train()
    va = data.load_validation()
    te = data.load_test()

    # figures
    f_npv = fig_npv_curve()
    f_time = fig_timing_hist(tr)
    f_sel = fig_selection(tr)
    f_miss = fig_missingness(tr)
    f_cum = fig_cumulative_default(tr)
    f_pipe = fig_pipeline()

    ss = _styles()
    S = []
    P = lambda t, st="Body": S.append(Paragraph(t, ss[st]))

    def bullets(items):
        S.append(ListFlowable(
            [ListItem(Paragraph(x, ss["BulletX"]), leftIndent=10, value="•") for x in items],
            bulletType="bullet", start="•"))

    # ---- Title ----
    S.append(Spacer(1, 1.4 * inch))
    P("SMB Underwriting Challenge", "TitleBig")
    S.append(Spacer(1, 6))
    P("Understanding the Data &amp; the End-to-End Logic", "Sub")
    S.append(Spacer(1, 10))
    P("Intuit TechWeek NYC AI/ML Hackathon — strategy &amp; data brief. "
      "Every number here is computed from the real dataset by <font face='Courier'>src/eda.py</font> "
      "and <font face='Courier'>src/report_pdf.py</font>.", "Small")
    S.append(PageBreak())

    # ---- 1. The problem ----
    P("1.  The problem in one paragraph", "H1b")
    P("You are a small-business lender. Using ~100K historical loan applications you must "
      "(<b>A</b>) decide whom to fund, (<b>B</b>) forecast how that funded book defaults over "
      "time, (<b>C</b>) answer causal “what-if” questions, and (<b>D</b>) defend it all in a "
      "4-page-max writeup. It is deliberately <b>not</b> a single-metric Kaggle task: the data bakes "
      "in real underwriting pathologies (selection bias, censoring, informative missingness, "
      "confounding) and the score rewards economic reasoning, calibration, and causal honesty.")
    P("Every loan, if funded, has fixed terms: amount = requested, term = 60 days repaid by daily "
      "ACH draws, APR = 35%, origination fee = 3%. A loan defaults on 3 consecutive missed draws, "
      "6 cumulative missed draws, or any balance left at day 90.")
    P("How the final score splits (drives where we spend time):", "H2b")
    S.append(_kv_table([
        ["Component", "Weight", "Deliverable", "What it measures"],
        ["Portfolio P&amp;L", "0.30", "A", "Realized profit of the loans you funded"],
        ["Trajectory", "0.25", "B", "Accuracy of cohort default-timing curve"],
        ["Calibration", "0.20", "A & B", "Do your 90% intervals contain truth, not too wide"],
        ["Causal", "0.10", "C", "Closeness to true interventional effects"],
        ["Writeup", "0.15", "D", "Quality of methodological defense"],
    ], [1.3 * inch, 0.7 * inch, 0.9 * inch, 3.5 * inch]))
    P("<b>Strategic read:</b> A + B + calibration are 75% of the score and A & B share one model. "
      "C is only 10% — keep it simple and defensible; the points live in the §3 writeup defense.", "Small")

    # ---- 2. The dataset ----
    P("2.  The dataset at a glance", "H1b")
    S.append(_kv_table([
        ["Frame", "Rows", "Cols", "Notes"],
        ["train", f"{len(tr):,}", "44", "Outcomes only for prior-approved + matured loans"],
        ["validation", f"{len(va):,}", "44", "Has outcomes — use to tune & calibrate"],
        ["test", f"{len(te):,}", "44", "Outcomes withheld — you are scored on these"],
    ], [1.1 * inch, 0.9 * inch, 0.6 * inch, 3.8 * inch]))
    P(f"You decide on validation + test combined = <b>{len(va)+len(te):,}</b> applicants (Deliverable A).")
    P("Features come in 8 groups: <b>business identity</b>, <b>self-reported</b> (applicant-stated), "
      "<b>bank-feed</b> (partial coverage), <b>bureau credit</b>, <b>platform engagement</b>, "
      "<b>application context</b>, <b>prior-underwriter</b> output, and <b>outcomes</b> "
      "(approved-only, must be dropped from inputs to avoid leakage).")

    # ---- 3. Five things to consider ----
    S.append(PageBreak())
    P("3.  Five things we must account for in the data", "H1b")
    P("These are the assumption-violations that make the problem hard — and the writeup ammunition.")

    P("3.1  Selective labels / sample-selection bias", "H2b")
    P("Repayment is observed <b>only</b> for the ~60.6% of loans a <i>prior</i> underwriter approved "
      "(51,722). The other 39.4% (33,618) were declined and are <b>unlabeled</b>. Train naively and "
      "you learn P(default | <i>approved</i>), not P(default | <i>applied</i>) — a biased view of the "
      "very applicants you must now newly decide on. The prior decision is near-deterministic in "
      "<font face='Courier'>prior_underwriter_score</font> (mean 0.781 approved vs 0.073 declined), so "
      "we model the propensity of approval and reweight (IPW) / report corrected vs uncorrected PD.")
    S.append(_img(f_sel))

    P("3.2  Censoring", "H2b")
    P("Among labeled loans, 42,698 paid in full and 9,024 defaulted (<b>17.4%</b> default rate). "
      "Paid-in-full loans are <i>right-censored</i> at repayment (they simply never defaulted); "
      "immature/declined loans carry no event. This is a survival problem, not plain classification — "
      "we model time-to-default and treat repayment as censoring.")

    P("3.3  Informative (MNAR) missingness", "H2b")
    P("Bank-feed columns are null exactly when no feed is linked (~35.7% of rows); “days since last "
      "decline/inquiry” is null when none occurred (~50%). The missingness is <b>not</b> random — its "
      "presence is itself a signal. We add explicit <font face='Courier'>*_is_missing</font> flags and "
      "sentinel-fill rather than imputing it away.")
    S.append(_img(f_miss))

    P("3.4  Optimistic self-report bias — but verify before claiming", "H2b")
    P("Self-reported fields could be inflated vs. the bank-feed ground truth. In <i>this</i> data the "
      "effect is <b>mild</b>: stated/observed monthly revenue has median ~ 0.98. Lesson: lean on "
      "observed bank-feed signals over stated ones, but don’t overstate the bias in the writeup — the "
      "data says applicants are roughly honest on average.")

    P("3.5  Confounding (for the causal deliverable)", "H2b")
    P("Deliverable C asks for P(default | <b>do</b>(feature = v)), not P(default | feature = v). These "
      "differ whenever the feature is confounded (e.g. revenue drives both utilization and default). "
      "A naive “change the column and re-predict” answers the wrong (observational) question; the "
      "rubric explicitly rewards interventional reasoning.")

    # ---- 4. Economics ----
    S.append(PageBreak())
    P("4.  The economics: why default <i>timing</i> drives everything", "H1b")
    P("The correct funding decision is <b>not</b> a flat PD threshold — it is <b>approve iff "
      "E[NPV] &gt; 0</b>. NPV depends on <i>when</i> a loan defaults, because the borrower pays a "
      "fixed daily draw until they stop:")
    P("&nbsp;&nbsp;• repaid: &nbsp; NPV = F + R*r*(T/365)<br/>"
      "&nbsp;&nbsp;• default at day t*: &nbsp; NPV = F + D*(t*-1) + recovery - R<br/>"
      "&nbsp;&nbsp;where F = 0.03R, D = R(1+rT/365)/T, r = 0.35, T = 60.", "Small")
    P("For a $10K loan this means a repaid loan earns <b>+$875</b>, a day-1 default loses "
      "<b>-$9,700</b>, but a day-60 default is actually <b>+$699</b> — almost the whole principal "
      "was already collected. <b>An early default and a late default look identical to a yes/no "
      "classifier but have a ~$10K swing in profit.</b> That is exactly why A, B, and the NPV "
      "decision all require a model of timing, not just probability.")
    S.append(_img(f_npv))
    S.append(Spacer(1, 4))
    S.append(_img(f_time))

    # ---- 5. Trajectory ----
    S.append(PageBreak())
    P("5.  The default trajectory (Deliverable B intuition)", "H1b")
    P("Deliverable B asks, for each origination cohort and loan age, the cumulative fraction of "
      "<i>your approved</i> loans that have defaulted by that age — a curve that can only rise. "
      "Below is the empirical cumulative-default curve over all approved loans; a real submission "
      "produces one such monotone curve per cohort (13×13 grid) with 90% bands. It comes directly "
      "from aggregating the shared hazard model, which is why B reuses A’s engine.")
    S.append(_img(f_cum))

    # ---- 6. End-to-end logic ----
    S.append(PageBreak())
    P("6.  End-to-end logic — one spine, four deliverables", "H1b")
    P("The architecture deliberately centers on a single <b>discrete-time hazard model</b> that "
      "outputs, per applicant, h(t) = P(default on day t | survived to t) for t = 1..90. Everything "
      "else is a read-off of that object:")
    bullets([
        "<b>A (decisions):</b> survival curve -&gt; P(default) and the timing distribution -&gt; expected NPV "
        "-&gt; approve iff E[NPV] &gt; 0. Also emit predicted PD + 90% interval.",
        "<b>B (trajectory):</b> aggregate the same per-day hazards over the approved loans in each "
        "cohort -&gt; monotone cumulative-default curve + band.",
        "<b>C (counterfactual):</b> reuse the feature representation but answer do(feature = v) with a "
        "causal estimator, contrasted against the naive re-prediction.",
        "<b>Selection-bias correction (IPW):</b> a propensity-of-approval model reweights the labeled "
        "loans so the hazard model reflects all applicants, not just previously-approved ones.",
        "<b>Calibration:</b> conformal / quantile methods tune the 90% intervals on held-out "
        "validation for ~90% coverage without being needlessly wide (0.20 of the score).",
        "<b>D (writeup):</b> defends each choice; §3 (causal) carries the most weight.",
    ])
    S.append(Spacer(1, 6))
    S.append(_img(f_pipe, width=6.4 * inch))

    # ---- 7. Per-deliverable cheat sheet ----
    S.append(PageBreak())
    P("7.  How each deliverable is produced", "H1b")
    S.append(_kv_table([
        ["", "Rows", "Core method", "Main gotcha"],
        ["A decisions", "13,306", "hazard -> NPV sign", "PD required even for declines; intervals ordered"],
        ["B trajectory", "169", "aggregate hazards by cohort", "must be monotone non-decreasing in age"],
        ["C counterfactual", "900", "causal do(), not re-predict", "confounding; ~16 intervenable features only"],
        ["D writeup", "<=4 pg", "narrative from A/B/C + EDA", "§3 causal weighted most; 11pt / 0.75in margins"],
    ], [1.25 * inch, 0.7 * inch, 1.95 * inch, 2.6 * inch]))
    P("<b>Hard gate:</b> exact file names, flat folder, and "
      "<font face='Courier'>validate_submission.py</font> must print PASS (it checks IDs, [0,1] "
      "ranges, lower &lt;= point &lt;= upper, the full 13×13 grid, and B monotonicity). A failing validator is "
      "disqualified, so we keep a passing baseline in <font face='Courier'>out/</font> at all times.")

    # ---- 8. Status ----
    P("8.  Where we are now (Phase 0 complete)", "H1b")
    bullets([
        "Reproducible scaffold in <font face='Courier'>src/</font> — data loaders, schema, NPV engine "
        "(self-test passes), submission writers, validator harness, EDA + this report generator.",
        "A complete, <b>validator-PASS</b> baseline submission sits in <font face='Courier'>out/</font> "
        "(A: 13,306, B: 169, C: 900) — we can never be in a non-submittable state.",
        "Next (Phase 1): feature pipeline (MNAR flags, ratios) + the discrete-time hazard model, "
        "compared across families on a common validation NPV metric.",
    ])
    P("One command reproduces Phase 0: <font face='Courier'>python run.py phase0</font>. "
      "This document: <font face='Courier'>python -m src.report_pdf</font>.", "Small")

    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=letter,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                            leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            title="SMB Underwriting — Data & Logic Overview")
    doc.build(S)
    print(f"PDF written to {PDF_PATH}")
    return str(PDF_PATH)


if __name__ == "__main__":
    build_pdf()
