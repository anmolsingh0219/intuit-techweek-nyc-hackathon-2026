"""Loan economics — NPV of a funded loan, exactly per the hackathon brief.

Product terms (fixed for every loan):
    APR  r = 0.35           term T = 60 days        origination fee F = 0.03 * R
    daily ACH draw  D = R * (1 + r*T/365) / T

NPV (profit, net of the principal R you lent out):
    repaid (y=0):              NPV = F + R * r * (T/365)
    default at day t* (y=1):   NPV = F + D * (t* - 1) + recovery - R

Key consequence the challenge hinges on: an EARLY default (small t*) is a large
loss; a LATE default (t* near 60) can still be ~break-even because most daily
draws were already collected. So the approve/decline decision needs default
*timing*, not just a yes/no PD. The economically-coherent policy is:

    approve  iff  E[NPV | approve] > 0
"""
from __future__ import annotations

import numpy as np

# Product constants
APR = 0.35
TERM_DAYS = 60
ORIG_FEE_RATE = 0.03
DAYS_PER_YEAR = 365


def origination_fee(R: np.ndarray | float) -> np.ndarray | float:
    return ORIG_FEE_RATE * np.asarray(R, dtype=float)


def daily_draw(R: np.ndarray | float) -> np.ndarray | float:
    """D = R * (1 + r*T/365) / T  — the fixed daily ACH amount."""
    R = np.asarray(R, dtype=float)
    return R * (1.0 + APR * TERM_DAYS / DAYS_PER_YEAR) / TERM_DAYS


def npv_repaid(R: np.ndarray | float) -> np.ndarray | float:
    """Profit if the loan is fully repaid: fee + interest."""
    R = np.asarray(R, dtype=float)
    return origination_fee(R) + R * APR * (TERM_DAYS / DAYS_PER_YEAR)


def npv_default(
    R: np.ndarray | float,
    t_star: np.ndarray | float,
    recovery: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """Profit if the loan defaults at day t*. (t*-1) daily draws were collected.

    Collected draws are capped at the 60-day term: a default flagged at the day-90
    window (t*=90) cannot have collected more than 60 daily draws, so we clamp
    (t*-1) at TERM_DAYS. For t* <= 60 this is exact.
    """
    R = np.asarray(R, dtype=float)
    t_star = np.asarray(t_star, dtype=float)
    recovery = np.asarray(recovery, dtype=float)
    collected = np.minimum(t_star - 1.0, float(TERM_DAYS))
    return origination_fee(R) + daily_draw(R) * collected + recovery - R


def realized_npv(
    R: np.ndarray,
    defaulted: np.ndarray,
    t_star: np.ndarray,
    recovery: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Vectorized realized NPV given known outcomes (for backtesting on val)."""
    R = np.asarray(R, dtype=float)
    defaulted = np.asarray(defaulted).astype(bool)
    t_star = np.asarray(t_star, dtype=float)
    recovery = np.broadcast_to(np.asarray(recovery, dtype=float), R.shape)
    out = npv_repaid(R)
    out = np.where(defaulted, npv_default(R, np.nan_to_num(t_star, nan=TERM_DAYS), recovery), out)
    return out


def expected_npv_from_hazard(
    R: np.ndarray,
    day_default_prob: np.ndarray,
    recovery_rate: float | np.ndarray = 0.0,
) -> np.ndarray:
    """E[NPV] for each loan from a per-day default distribution.

    Parameters
    ----------
    R : (n,)            requested amounts.
    day_default_prob : (n, T)  P(default exactly on day t), t = 1..T (T==TERM_DAYS).
        Row sum = P(default within term) <= 1; the remainder is P(repaid).
    recovery_rate : recovered fraction of principal on default (0 = none).

    Returns E[NPV] per loan, integrating the timing-dependent default payoff.
    """
    R = np.asarray(R, dtype=float).reshape(-1, 1)
    P = np.asarray(day_default_prob, dtype=float)
    if P.shape[1] != TERM_DAYS:
        raise ValueError(f"day_default_prob must have {TERM_DAYS} day-columns, got {P.shape[1]}")
    t = np.arange(1, TERM_DAYS + 1, dtype=float).reshape(1, -1)
    recovery = np.asarray(recovery_rate, dtype=float) * R  # broadcast (n,1)

    npv_def_t = origination_fee(R) + daily_draw(R) * (t - 1.0) + recovery - R  # (n,T)
    p_default_total = P.sum(axis=1, keepdims=True)
    p_repaid = np.clip(1.0 - p_default_total, 0.0, 1.0)

    e_npv = (P * npv_def_t).sum(axis=1, keepdims=True) + p_repaid * npv_repaid(R)
    return e_npv.ravel()


def expected_npv_weekly(
    R: np.ndarray,
    pi_weeks: np.ndarray,
    day_per_week: np.ndarray,
    recovery_rate: float = 0.0,
) -> np.ndarray:
    """E[NPV] per loan from a weekly unconditional default distribution.

    Parameters
    ----------
    R : (n,)               requested amounts.
    pi_weeks : (n, K)      pi[i,k] = P(loan i defaults in default-week k).
                           Row sum = PD_i (<=1); remainder 1-PD_i is P(repaid).
    day_per_week : (K,)    representative default day for each week (e.g. the
                           empirical mean days_to_default within that week).
    recovery_rate : recovered fraction of principal on default.

    E[NPV] = (1-PD)*npv_repaid + sum_k pi_k * npv_default(day_k, rec_rate*R).
    """
    R = np.asarray(R, dtype=float)
    pi = np.asarray(pi_weeks, dtype=float)
    days = np.asarray(day_per_week, dtype=float).reshape(1, -1)
    if pi.shape[1] != days.shape[1]:
        raise ValueError("pi_weeks and day_per_week must share the week dimension")
    Rc = R.reshape(-1, 1)
    rr = np.asarray(recovery_rate, dtype=float)
    rec = (rr.reshape(-1, 1) if rr.ndim else rr) * Rc   # per-loan array or scalar
    npv_def = origination_fee(Rc) + daily_draw(Rc) * np.minimum(days - 1.0, float(TERM_DAYS)) + rec - Rc
    pd_total = pi.sum(axis=1)
    p_repaid = np.clip(1.0 - pd_total, 0.0, 1.0)
    return (pi * npv_def).sum(axis=1) + p_repaid * npv_repaid(R)


def _selftest() -> None:
    R = 10_000.0
    D = daily_draw(R)
    assert abs(D - 10_000 * (1 + 0.35 * 60 / 365) / 60) < 1e-9
    # Repaid: fee + interest.
    assert abs(npv_repaid(R) - (300.0 + 10_000 * 0.35 * 60 / 365)) < 1e-6
    # Default at t*=1 (immediate): lose almost everything.
    assert abs(npv_default(R, 1) - (300.0 - 10_000)) < 1e-6
    # Default at the last day ~ recovers most via collected draws.
    assert npv_default(R, 60) > 0
    # Monotonic in timing: later default never worse than earlier.
    ts = np.arange(1, 61)
    vals = npv_default(R, ts)
    assert np.all(np.diff(vals) > 0)
    # Hazard integration sanity: a loan that always repays == npv_repaid.
    P0 = np.zeros((1, 60))
    assert abs(expected_npv_from_hazard(np.array([R]), P0)[0] - npv_repaid(R)) < 1e-6
    # A loan that defaults for sure on day 1 == npv_default(t=1).
    P1 = np.zeros((1, 60)); P1[0, 0] = 1.0
    assert abs(expected_npv_from_hazard(np.array([R]), P1)[0] - npv_default(R, 1)) < 1e-6
    print("npv self-test: PASS")
    print(f"  D={D:.4f}  repaid={npv_repaid(R):.2f}  "
          f"def@1={npv_default(R,1):.2f}  def@30={npv_default(R,30):.2f}  "
          f"def@60={npv_default(R,60):.2f}")


if __name__ == "__main__":
    _selftest()
