# Three-Way Submission Grading — Anmol vs Aryan vs Franco

*Scored on the 2,551 ground-truthed validation loans (the only rows with labels). Same NPV
formula applied to every team's `decision` column, so the P&L comparison is apples-to-apples.
Test P&L is held by the organizers and cannot be computed by us.*

## Objective scoreboard

### Deliverable A (decisions + PD + 90% intervals)
| team | approve | realized val P&L | AUC | Brier | cal-gap | A interval (width @ coverage) |
|---|---|---|---|---|---|---|
| **Anmol (us)** | 0.742 | $2,099K | **0.7589** | **0.1340** | +0.000 | **0.080 @ 1.00** |
| **Aryan** | 0.741 | **$2,180K** | 0.7549 | 0.1347 | +0.000 | 0.131 @ 0.90 |
| **Franco** | 0.740 | $2,001K | 0.7483 | 0.1356 | −0.000 | 0.102 @ 1.00 |

### Deliverable B (169 cohort × age cells)
| team | monotone | week-13 level | interval width |
|---|---|---|---|
| Anmol | 100% | 0.132 | **0.033 (binomial, ~91% val coverage)** |
| Aryan | 100% | 0.130 | **0.002 — recklessly narrow** |
| Franco | 100% | 0.136 | 0.047 (safe / wide) |

### Deliverable C (900 counterfactual queries)
| team | mean PD | mean interval width | degenerate [0,1] | lower-clipped-at-0 |
|---|---|---|---|---|
| Anmol | 0.314 | 0.177 | 0 | 106 |
| Aryan | 0.265 | 0.266 | **6** | 49 |
| Franco | 0.283 | 0.185 | 0 | 0 |

## Reasoning, dimension by dimension

**P&L (30%).** All three converged on the high-approval policy (~74% on the labeled set), so the
earlier 2.5× gap is gone. On realized val P&L **Aryan leads ($2,180K), us second ($2,099K),
Franco third ($2,001K)**. Aryan's edge (+$81K at the *same* approve count) comes from a
threshold he explicitly "tuned on realized validation profit" plus a PD-tercile LGD — i.e. likely
*val-overfit* that may not transfer to test. Our AUC is the highest (0.759), which is the most
transfer-robust ranking, and our τ is principled (floored at 0, not leaderboard-fit). Net: Aryan wins
the measurable number; we have the better test-transfer case. Franco's policy is the most
principled (threshold from economics, "not tuned to the leaderboard," one-line flip if the scorer
credits no draws) but realizes the least profit.

**Trajectory B (25%).** All monotone; week-13 levels nearly identical (~0.13). The differentiator is
the interval. Ours is binomial-sized to ~91% val coverage (width 0.033). Franco's is competing-risks
and a touch wide (0.047). **Aryan's is 0.002 wide — essentially a point estimate**; if B coverage is
scored, realized cohort rates fall outside almost every cell and he takes a large penalty. High-variance bet.

**Calibration (20%) — our strongest, objectively.** Lowest Brier (0.134), sharpest A intervals at
*full* coverage (0.080 vs 0.10–0.13), best-calibrated B. Aryan's A intervals are widest *and* lowest
coverage (worst-shaped); his B intervals are dangerously narrow. Franco is solid with a split-conformal
guarantee but wider (0.102).

**Counterfactual C (10%) — we are SECOND, behind Franco.** Our *file* is fine (bounded HGB, 0
degenerate bands, tightest width). Two gaps vs Franco: (1) **106 of our lower bounds clip to 0** — a
flat ±floor band is uninformative for low-PD queries, where Franco's shaped intervals never clip; (2)
Franco's C *reasoning* is more rigorous — he bounds confounded features with **Manski + RD**
rather than just reporting the model response. Aryan is third on C: 6 degenerate [0,1] bands, widest
intervals, 39 ceiling-hits.

**Writeup D (15%).** Honest call: **ours is the best-written but the least rigorous of the three.**
Franco's has a cumulative-gains chart (KS 0.37, Gini 0.50), a McCrary density test, executed Manski
bounds [0.11, 0.50], and a full academic reference list — the most complete and professional. Aryan's
is dense and specific (the 0.273 threshold, Platt 0.4308 vs isotonic 0.4674, a do(stated_revenue)=+0.016
effect) with a sophisticated §5 (Manski, RD-as-instrument, double-ML, competing-risks per trigger). The
RD anchor and two-mode timing we leaned on are *already in both of their writeups*. Our edge is voice
and the RD-validation numbers (23%→25%→32%); our gap is formal rigor (no McCrary/Manski/refs).

## Grades (1–10, weighted by the rubric S = .30·P&L + .25·traj + .20·cal + .10·C + .15·write)

| Dimension | weight | Anmol | Aryan | Franco |
|---|---|---|---|---|
| P&L | 30% | 8.0 | **9.0** | 7.5 |
| Trajectory B | 25% | **8.5** | 6.0 | 8.0 |
| Calibration | 20% | **9.0** | 6.0 | 8.0 |
| Counterfactual C | 10% | 8.0 | 7.0 | **8.5** |
| Writeup D | 15% | 8.5 | 8.5 | **9.0** |
| **Weighted total** | | **8.40** | **7.38** | **8.05** |

## Verdict (not flattering)

**Near-tie between us and Franco (8.40 vs 8.05); Aryan third (7.38) but highest-variance.** We lead on
*balance and calibration discipline*; Franco leads on *rigor and presentation* and is the most professional
submission; Aryan owns *raw val P&L* but is fragile (0.002 B intervals, 6 degenerate C bands).

The order flips on two unknowns: (a) if test P&L rewards Aryan's val-tuned policy, he takes the 30%
bucket and likely #1; (b) if interval coverage is lightly weighted, our calibration edge shrinks and Franco's
superior writeup pulls level. We do **not** dominate — the field converged, and our advantage is now in the
details.

## What we are doing about it
1. **Writeup rigor** — add the McCrary density check, Manski bounds, and a reference list (close the Franco gap on 15%).
2. **Counterfactual C** — reshape intervals to Bernoulli (√p(1−p)) form so low-PD queries stop clipping at 0 (match Franco's clean distribution); strengthen the §3 causal bounds.
3. **P&L** — tested honestly; our pipeline already tunes τ and floors it at 0 to avoid the exact val-overfit that produces Aryan's lead. Chasing the $81K means removing that guard, which hurts test. We keep the principled τ.
