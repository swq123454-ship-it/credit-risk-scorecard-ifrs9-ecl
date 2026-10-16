"""Financial mathematics shared by the data generator and the IFRS 9 engine.

Keeping these primitives in one auditable module matters: the amortisation
schedule, the marginal PD term structure and the discount factors are the parts
of an ECL calculation that auditors test line by line.
"""

from __future__ import annotations

import numpy as np


def annuity_payment(
    principal: np.ndarray | float, annual_rate_pct: np.ndarray | float, months: np.ndarray | int
) -> np.ndarray:
    """Level payment that fully amortises ``principal`` over ``months``.

    ``annual_rate_pct`` is a nominal annual percentage rate (e.g. ``12.5`` for
    12.5%), consistent with how retail loan rates are quoted in the source data.
    """
    principal = np.asarray(principal, dtype=float)
    months = np.asarray(months, dtype=float)
    monthly_rate = np.asarray(annual_rate_pct, dtype=float) / 100.0 / 12.0

    with np.errstate(divide="ignore", invalid="ignore"):
        payment = np.where(
            monthly_rate <= 0,
            principal / np.maximum(months, 1.0),
            principal * monthly_rate / (1.0 - np.power(1.0 + monthly_rate, -months)),
        )
    return payment


def outstanding_balance(
    principal: np.ndarray | float,
    annual_rate_pct: np.ndarray | float,
    months_total: np.ndarray | int,
    months_elapsed: np.ndarray | int,
) -> np.ndarray:
    """Remaining principal after ``months_elapsed`` payments.

    Uses the standard closed-form annuity balance, which is algebraically
    identical to walking the amortisation schedule month by month but runs in
    constant time per loan.
    """
    principal = np.asarray(principal, dtype=float)
    months_total = np.asarray(months_total, dtype=float)
    months_elapsed = np.asarray(months_elapsed, dtype=float)
    monthly_rate = np.asarray(annual_rate_pct, dtype=float) / 100.0 / 12.0

    elapsed = np.clip(months_elapsed, 0.0, months_total)
    remaining_months = months_total - elapsed
    payment = annuity_payment(principal, annual_rate_pct, months_total)

    with np.errstate(divide="ignore", invalid="ignore"):
        balance = np.where(
            monthly_rate <= 0,
            np.maximum(principal - payment * elapsed, 0.0),
            payment * (1.0 - np.power(1.0 + monthly_rate, -remaining_months)) / monthly_rate,
        )
    balance = np.where(remaining_months <= 0, 0.0, balance)
    return np.clip(balance, 0.0, principal)


def monthly_hazard_from_annual_pd(pd_12m: np.ndarray | float) -> np.ndarray:
    """Convert a 12-month PD into a constant monthly hazard rate.

    ``h`` solves ``1 - (1 - h) ** 12 = PD_12m``. This is the standard
    "constant hazard" term structure used when a lifetime PD curve is not
    available.
    """
    pd_12m = np.clip(np.asarray(pd_12m, dtype=float), 1e-9, 1.0 - 1e-9)
    return 1.0 - np.power(1.0 - pd_12m, 1.0 / 12.0)


def marginal_pd_curve(
    pd_12m: np.ndarray | float,
    horizon_months: int,
) -> np.ndarray:
    """Unconditional monthly default probability over ``horizon_months``.

    Returns an array of shape ``(n_loans, horizon_months)`` where entry
    ``(i, t)`` is the probability that loan ``i`` defaults in month ``t + 1``
    given that it had not defaulted before. Marginal (unconditional) PDs are
    what get multiplied by EAD and discounted, not the conditional hazard.
    """
    hazard = monthly_hazard_from_annual_pd(pd_12m)
    hazard = np.atleast_1d(hazard).reshape(-1, 1)
    steps = np.arange(horizon_months).reshape(1, -1)
    survival = np.power(1.0 - hazard, steps)
    return survival * hazard


def discount_factors(annual_rate: float, horizon_months: int) -> np.ndarray:
    """Month-end discount factors for a nominal annual rate."""
    monthly_rate = annual_rate / 12.0
    steps = np.arange(1, horizon_months + 1)
    return 1.0 / np.power(1.0 + monthly_rate, steps)


def cumulative_lifetime_pd(pd_12m: np.ndarray | float, horizon_months: int) -> np.ndarray:
    """Cumulative probability of default over ``horizon_months``."""
    hazard = monthly_hazard_from_annual_pd(pd_12m)
    return 1.0 - np.power(1.0 - hazard, horizon_months)
