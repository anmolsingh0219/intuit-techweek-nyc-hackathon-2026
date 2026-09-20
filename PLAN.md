# Solo Battle Plan — SMB Underwriting Challenge

> Deadline: **Saturday 14:00**. Solo. Goal: maximize
> `S = 0.30·P&L + 0.25·Traj + 0.20·Cal + 0.10·Causal + 0.15·Writeup`.
> Strategy: A and B share ONE survival model; calibration is cheap points;
> C is small (don't over-invest); writeup Section 3 is the causal centerpiece.

---

## North-star decisions (locked)

1. **One model spine for A + B.** Build a *discrete-time hazard* model that gives,
   per applicant, `h(t)` = P(default on day t | survived to t), for t = 1..90.
   Everything else derives from it:
   - `predicted_pd` (A) = `1 - Π(1 - h(t))` over t=1..90.
   - default-timing distribution → expected `t*` → **NPV** → approve/decline (A).
   - cohort cumulative-incidence curve (B) by aggregating approved loans.
   This is what makes "day-5 vs day-55 default" distinguishable, which A+B+NPV all need.

2. **Decision rule = `approve iff E[NPV] > 0`**, NOT a flat PD threshold.
   NPV uses the timing from the hazard model. This is the economically-coherent
   policy the rubric rewards.

3. **Fix selection bias explicitly.** Labels exist only for prior-APPROVED loans
   (51,722 of 85,340; 39% declined & unlabeled). Don't ignore it — at minimum do
   IPW (inverse propensity of approval) or a 2-stage/Heckman-style correction,
   and *write about it*. This is assumption-violation gold for Section 1 & 3.

4. **Treat missingness as signal (MNAR).** Bank-feed cols null ⟺ no linked feed.
   Add explicit `*_is_missing` flags + sentinel fill; never silently impute away.

5. **Calibrate on held-out validation** via conformal / quantile methods for the
   90% intervals on A, B, C. Coverage ≈ 90% without absurd width = the 0.20 bucket.

---

## Phase 0 — Setup & EDA  (Fri evening, ~2 hr)  ✅ DONE — `python run.py phase0`

- [ ] Register on the Google Form **before 8PM Friday** (gets submission link). ← USER must do
- [x] Repo scaffold: `src/` (paths, data, npv, submission, validate, eda), `out/`,
      `reports/`, `experiments/`, `run.py` orchestrator.
- [x] `data.py`: zip loaders, schema parsed from data_dictionary (feature/outcome/
      intervenable/MNAR-nullable groups), survival-label builder, approval indicator.
- [x] `npv.py`: NPV exactly from brief + `expected_npv_from_hazard`; self-test PASS
      (D=176.26, repaid=875.34, def@1=-9700, def@60=+699 → timing dominates).
- [x] EDA report → `reports/eda_report.md` (selection bias, censoring, MNAR map,
      self-report ratio, timing). Leakage handled: OUTCOME_COLS excluded from features.
- [x] `validate.py` harness + `submission.build_dummy()` → out/ passes validator NOW.

**Exit criterion:** ✅ validator prints PASS on dummy submission; NPV fn tested.

**Phase-0 findings to carry forward:**
- `prior_underwriter_score`: approved mean **0.781** vs declined **0.073** → near-
  deterministic selection mechanism; use as propensity-model backbone for IPW.
- Self-report bias is **mild** (stated/observed monthly rev median 0.98) — don't over-claim.
- ~10% of defaults land at the day-90 window (`days_to_default` p90 = 90). **Phase-1 TODO:**
  `expected_npv_from_hazard` currently models 60 day-bins; handle late (61–90d) defaults
  explicitly (they are NOT "repaid"), and cap collected draws at T=60 in realized NPV.

---

## Phase 1 — Core risk model A+B  ✅ DONE — `python run.py phase1`

**Built:** `features.py` (XGB-native NaN + MNAR flags + ratio features),
`cohorts.py` (timestamp→week), `survival.py` (two-stage **PD × timing** model,
bagged PD intervals, IPW propensity), `evaluate.py` (PD metrics + realized policy
P&L), `phase1.py` orchestrator. Writes real A & B → validator **PASS**.

**Why two-stage (not full survival):** labels are *fully observed* (all matured;
repayment deterministic at day 60; no censoring), and default-week support is
{1–9, 13}. So PD = P(default by 90) [binary, bagged] × timing = P(week | default)
[multiclass over observed weeks]. pi = PD·w, CIF = cumsum(pi), E[NPV] via weekly
draws. Clean, exact, monotone-by-construction, calibratable.

**Results (validation):**
- PD: AUC **0.753**, Brier 0.135, slight under-prediction (calib_gap −0.02).
- **Economic policy wins:** NPV>0 realized P&L **$2.61M** vs approve-all $0.61M
  (4.3×) and beats every flat PD threshold on *total* P&L → confirms approve-iff-
  E[NPV]>0. Approve rate 0.77 on labeled val, **0.657 on full pool** (riskier).
- Our approved book defaults ~12% vs 17.4% overall approved → we select profitably.
- **IPW ≈ no-op:** approved loans have propensity≈1 (poor overlap with declined,
  positivity violated), so weights collapse to ~constant; plain == IPW. Honest
  limitation for §3 writeup (selection correction is fundamentally limited here).

**Improvements A+B+C (done — `models.py` + upgraded `phase1.py`):**
- **A. Monotone + robust:** XGB/LGBM/CatBoost/HGB get regulator-defensible
  monotonic constraints (more debt/util/inquiries → never less risk; more
  revenue/cash → never more) + heavier regularization, so extrapolation into the
  label-scarce risky region behaves sanely.
- **B. Tuned approval buffer τ:** approve iff E[NPV] > τ, τ chosen on val P&L.
  For the well-calibrated logistic model τ=$0 is already optimal; tree models
  wanted τ>0 (they're optimistic). τ doubles as a safety diagnostic.
- **C. Bake-off (val P&L, AUC):** logistic **0.7570 / $2.69M** > catboost
  0.7525/$2.65M > hgb 0.7537/$2.65M > xgb 0.7499/$2.63M > lgbm 0.7473/$2.51M.
  **Logistic regression WON on both AUC and P&L** — the linear/inherently-smooth
  model extrapolates into the risky region better than trees that overfit the
  safe region. Ensemble (logistic+catboost) did *not* beat single logistic →
  kept it simple. **Final = bagged logistic**, val P&L **$2.694M** (vs $2.61M
  pre-improvement), AUC 0.758, approve 0.645 on full pool. Validates PASS.
- This is the real answer to "IPW didn't work": fix selection-region
  extrapolation via model choice + monotonicity, not reweighting. Great §3 story.

**Phase 1.5 (closed two real gaps — `features.py` v2 + timing validation):**
- **Encoding fix:** one-hot the nominal categoricals (sector/geography/use/channel
  — codes carry no order, fatal for the *linear* winner) + a few interaction/
  squared terms. Lifted every model; logistic AUC 0.757→**0.759**, P&L $2.69M→**$2.71M**.
- **Timing finally measured (Deliverable B = 0.25 of score, previously untested):**
  trajectory MAE **~1.5%** — excellent. Confirmed timing *is* x-dependent (high-util
  loans default ~2 weeks earlier, fewer reach the day-90 bucket), so **A's NPV
  decision uses conditional (per-loan) timing**; **B's cohort curve uses marginal**
  (slightly better aggregate MAE 0.0149 vs 0.0154). Decoupled the two.
- **τ floored at $0** (no approving −EV loans — killed a val-overfit where the
  ensemble looked better only by setting τ=−$200). With the floor, single logistic
  cleanly wins; ensemble only adopted if it beats single by >0.3%.
- **Final Phase-1 model: bagged single logistic**, AUC **0.759**, val P&L **$2.71M**,
  approve 0.652, B-traj-MAE ~1.5%. Validates PASS. Cached to phase1_preds.npz.

**Phase-1 TODOs → Phase 2/3:**
- Intervals are bag-heuristic (mean width 0.057 for A); **Phase 2 conformal**
  must calibrate to ~90% coverage on held-out val (the 0.20 score bucket).
- Day-90 (week-13) NPV uses draws capped at 60 + 9.1% recovery — document the
  assumption; consider sensitivity.
- Predictions cached to `experiments/_cache/phase1_preds.npz` for Phase 2/3 reuse.

### (original Phase 1 plan notes)  (Fri night → Sat AM, biggest block ~5–6 hr)

Build the feature pipeline once, then **experiment with multiple model families**
behind a common interface so you can swap and compare on the same val split.

- [ ] `features.py`: MNAR flags, categorical encoding, self-report/observed ratios,
      log-transform skewed dollar amounts, ratio features (req_amt / observed_rev).
      Drop all `outcome` group cols from X.
- [ ] Define the label for survival: event = default, time = `days_to_default`,
      censoring = paid_in_full (censored at repayment) / immature (censored at obs).
- [ ] **Experiment track (same val metric: time-aware Brier / C-index + NPV on val):**
      - Baseline A: XGBoost binary classifier on default_flag (sanity / fallback).
      - Survival 1: discrete-time hazard via gradient boosting (per-day or per-week
        hazard; pool over time bins 1..13 weeks).
      - Survival 2: scikit-survival (Cox / Random Survival Forest) or lifelines.
      - Survival 3 (stretch): parametric (Weibull AFT) for smooth trajectories.
- [ ] Apply **selection-bias correction**: fit P(prior_approve | x) propensity;
      reweight training (IPW) and/or compare corrected vs uncorrected PD on val.
- [ ] Pick winner by **val NPV of the resulting approve/decline policy** (that's
      what P&L scores), tie-broken by calibration.
- [ ] Produce A: predicted_pd + NPV decision. Produce B: aggregate chosen model's
      per-day hazard over approved test loans, bucket by cohort_week (use
      cohort_week_definitions + application_timestamp), enforce monotonicity.

**Exit criterion:** A and B numbers generated from one model; validator PASS;
val NPV beats the naive-threshold baseline.

---

## Phase 2 — Calibration & UQ  ✅ DONE — `python run.py phase2`  (← 0.20)

**Built:** `calibrate.py` (isotonic/Platt PD calibration + method selection by
held-out Brier; one-parameter conformal intervals for A & B), `phase2.py`.

**Results (held-out validation):**
- **PD calibration:** method auto-picked **Platt** (beat isotonic on held-out
  Brier). Brier 0.1344→**0.1338**, calib gap −0.025→**−0.008** (nearly closed).
- **A intervals:** k=0.10, floor=0.03, **held-out coverage 0.90** (on target),
  mean width 0.078, heteroscedastic (0.036–0.100; narrow at extremes).
- **B intervals:** k=1.53, **val cohort/age coverage 0.91**, mean width 0.032.
- Validates PASS.

**⚠ Tension flagged for review:** calibrated PD (correctly ~2.5pp higher, fixing
under-prediction) makes the NPV decision slightly more conservative — pool approve
0.652→**0.615**, val P&L $2.71M→$2.68M. The val P&L drop is the approved-only
*optimism artifact*; on the true (riskier) test population, more-accurate PD →
more-correct folds → should be ≥ uncalibrated. **RESOLVED (user call):** use calibrated PD for BOTH the reported probability and
the approve/decline decision — coherent, and more-accurate folds should win on the
true (riskier) test population despite the optimistic-val P&L dip.

### (original Phase 2 plan)  (Sat AM, ~2 hr)  ← 0.20, high ROI

- [ ] A intervals: split-conformal or quantile bins on val; widen PD to hit ≈90%
      empirical coverage. Verify `lower ≤ pd ≤ upper` and coverage on val labels.
- [ ] B intervals: bootstrap cohorts / Wilson interval on the cumulative fraction;
      keep band monotone-consistent.
- [ ] C intervals: propagate model uncertainty (ensemble spread / conformal).
- [ ] Log realized coverage & mean width for the writeup (Section 4).

**Exit criterion:** intervals ~90% coverage on val, not absurdly wide; validator PASS.

---

## Phase 3 — Causal / Counterfactual C  ✅ DONE — `python run.py phase3`  (← 0.10)

**Built:** `causal.py` (structural counterfactual + naive baseline), `pipeline.py`
(shared calibrated-model builder, backbone-configurable), `phase3.py`.

**Method:** `P(default | do(X=v))` = set raw feature → **rebuild feature vector**
(our pipeline IS the deterministic-propagation mechanism: engineered ratios,
interactions, one-hots, and the provided requested/observed ratio all recompute) →
calibrated bagged PD. Naive "patch one column" leaves descendants stale; we contrast.

**Key fix during build:** the **logistic A/B model is wrong for C** — being linear
it extrapolates a single feature change without bound (PD swinging 0.20→0.89, mean
|effect| 0.18). Switched C to **bagged monotone HistGB** (locally bounded, NaN-native
for no-feed applicants): mean |effect| **0.045** (credible ~4.5pp), mixed-sign,
directionally sensible (more overdrafts→riskier; revenue↓→riskier). Verified
mechanics with a no-op-intervention test (reproduces baseline exactly).

**Output:** 900 rows, PD∈[0.05,0.99], intervals (A conformal k/floor + 0.03 causal
widening) mean width 0.130. Validates PASS. Cached to phase3_preds.npz.

**Writeup §3 hooks:** observational≠interventional; deterministic propagation done
exactly, confounding NOT claimed de-confounded (monotone = weak causal prior, wider
intervals for residual uncertainty); model choice (bounded tree) for credible
magnitudes; some queries intervene on structurally-missing/identity features (ill-posed,
answered mechanically). NOTE: naive-baseline diagnostic mis-handles nominal one-hots
(spurious propagation# for geography/etc.) — cite propagation evidence from the
*continuous* features (revenue, requested_amount) which is correct.

### (original Phase 3 plan)  (Sat AM/noon, ~1.5 hr)  ← only 0.10

Keep it **defensible and simple**; the points are mostly in the *defense* (D §3).

- [ ] Sketch a DAG over the ~16 intervenable features → default. Identify obvious
      confounders (e.g. revenue → both utilization & default).
- [ ] Approach (pick one, defend it): T-learner / causal forest (econml) for the
      intervened feature, OR structural: model the feature's downstream children
      and recompute. Contrast with naive "set & re-predict" and state what naive gives up.
- [ ] Generate 900 predicted_pd_cf + intervals. Clip to [0,1].

**Exit criterion:** 900 rows, validator PASS, C method writable in 1 paragraph.

---

## Phase 5 — Rigorous optimization  ✅ DONE — `python run.py phase5` (+ analysis)

Goal: maximize P&L. Built `cv.py` (GroupKFold-by-business OOF, stacking, recovery
model), `phase5.py`, richer credit-risk features (73 total). Findings (rigorous):
- **TEMPORAL DRIFT is the key structural fact** (see dataset-key-facts memory):
  train = older/lower-default era (15%), val+test = recent/higher era (21%). It is
  **label shift, not concept shift** — feature→risk ranking is stable (recency
  weighting does NOT help; it hurts val). Correct fix = calibrate PD on the TARGET
  era (val) — exactly Phase 2. Train-OOF AUC 0.776 is era-optimism; **test-era AUC
  ceiling ≈ 0.758**. Logistic generalizes across the drift; trees overfit train era.
- Phase-5's OOF-calibration was a *regression* for the drift (train-era base rate)
  → reverted to Phase-2 val-calibration as the final.
- **Final config = logistic + 73 richer features + val-era Platt calibration +
  recovery model tested + E[NPV]>0 decision.** Re-ran phase2/phase3 to lock it in.
- **Decision policy proven optimal:** E[NPV]>0 ($2.67M val) beats a direct
  profit-classifier ($2.37M) and a direct NPV-regressor ($2.63M). Magnitude-aware
  NPV folds the −$9,700 early-defaulters specifically.
- **Oracle ceiling (perfect foresight) = $5.01M on val; we capture 53%** — the gap
  is irreducible model error at the 0.758 AUC ceiling, not a fixable mistake.
- **Head-to-head vs comparison team (2,551 ground-truthed val loans): we win** —
  P&L $2.67M vs $1.07M (2.5×), AUC 0.759 vs 0.748, Brier 0.134 vs 0.135, intervals
  0.080 vs 0.105 (tighter at equal coverage). Their flaw: over-conservative (approve
  15.6%). NOTE: $2.67M is the val *subset*; scored test P&L (8,817 loans) is larger.

**Conclusion: A/B are near the achievable ceiling for this data. Biggest remaining
points are the WRITEUP (0.15), which now has a rigorous, honest story.**

## Phase 4 — Writeup D  ✅ DONE — `python -m src.writeup_pdf`  (← 0.15)

`src/writeup_pdf.py` -> `out/submission_D_writeup.pdf`. **3 pages** (under the 4-page
cap), 11pt, 0.85" margins, all 5 required sections in order. Substantive, rigorous:
§1 leads with the temporal-drift/label-shift discovery + selection bias + MNAR;
§2 the PD×timing spine + E[NPV] decision (proven > profit-clf/NPV-reg) + logistic
bake-off win + monotone constraints; §3 (most-weighted) do() vs observational,
exact deterministic propagation via the feature pipeline, bounded-tree for
counterfactual magnitudes, confounding stance + regulator defense; §4 target-era
calibration (Platt by held-out Brier) + conformal A/B intervals (cov 0.90/0.91);
§5 honest limitations (day-90 NPV, positivity, val optimism, AUC ceiling, oracle 53%).
**ACTION: edit TEAM = "[your team name]" in writeup_pdf.py before final submit.**

## ===== ALL FOUR DELIVERABLES DONE — validator PASS (0 errors, 0 warnings) =====
out/: submission_A_decisions.csv, submission_B_trajectory.csv,
      submission_C_counterfactuals.csv, submission_D_writeup.pdf. Ready to upload.

### (original Phase 4 plan)  (Sat noon, ~1.5 hr, parallelize notes throughout) ← 0.15

Fill `submission_D_writeup_template.md`, export PDF (≤4pg, ≥11pt, ≥0.75" margin).
Section weights to remember: **§3 causal is weighted most.**
1. Framing & assumptions violated — selection bias, censoring, MNAR, self-report,
   confounding (you have receipts from EDA).
2. Methodology — the shared survival spine, NPV decision, selection correction.
3. **Causal** — do(X) vs P(y|X), your estimator, confounders, regulator defense.
4. Calibration — conformal method, held-out coverage/width tradeoff (real numbers).
5. Limitations — what you'd do with another day.

---

## Phase 5 — Final assembly & submit  (before 14:00, buffer ≥45 min)

- [ ] All 4 files exact-named, flat folder, run validator → **PASS**.
- [ ] Re-check ID coverage (13,306 applicants, 900 queries, 169 grid rows).
- [ ] Upload via the team's private link. Done.

---

## Time budget (deadline Sat 14:00)
| Block | When | Hrs |
|---|---|---|
| P0 setup+EDA+register | Fri 17:00–19:30 | 2.5 |
| P1 core A+B (+ overnight runs) | Fri 19:30 → Sat 09:30 | core ~5 active |
| P2 calibration | Sat 09:30–11:00 | 1.5 |
| P3 causal C | Sat 11:00–12:15 | 1.25 |
| P4 writeup | Sat 12:15–13:15 | 1.0 |
| P5 assemble+submit | Sat 13:15–14:00 | 0.75 buffer |

## Risk guards
- **Always keep a passing submission in `out/`** (start with the dummy, replace
  pieces as they're ready). Never be in a state where you can't submit.
- Timebox experiments: if a survival lib fights you for >45 min, fall back to the
  discrete-hazard-via-boosting track and move on.
- Validator is the gate — run it after every deliverable changes.
