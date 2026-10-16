"""Tests for the financial maths primitives.

These are the functions an auditor would re-perform by hand, so they get exact
expected values rather than smoke assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

from creditrisk.finmath import (
    annuity_payment,
    cumulative_lifetime_pd,
    discount_factors,
    marginal_pd_curve,
    monthly_hazard_from_annual_pd,
    outstanding_balance,
)


def test_annuity_payment_matches_closed_form():
    # 10,000 at 12% nominal over 36 months -> well-known 332.14 monthly payment.
    payment = annuity_payment(10_000.0, 12.0, 36)
    assert payment == pytest.approx(332.14, abs=0.01)


def test_annuity_payment_zero_rate_is_straight_line():
    assert annuity_payment(1_200.0, 0.0, 12) == pytest.approx(100.0)


def test_outstanding_balance_starts_at_principal_and_ends_at_zero():
    at_start = outstanding_balance(10_000.0, 12.0, 36, 0)
    at_end = outstanding_balance(10_000.0, 12.0, 36, 36)
    assert at_start == pytest.approx(10_000.0, rel=1e-6)
    assert at_end == pytest.approx(0.0, abs=1e-6)


def test_outstanding_balance_decreases_monotonically():
    balances = [outstanding_balance(10_000.0, 12.0, 36, month) for month in range(0, 37)]
    # strict=False on purpose: this is a sliding pair over the same series, so the
    # lengths differ by exactly one by construction.
    assert all(later <= earlier + 1e-9 for earlier, later in zip(balances, balances[1:], strict=False))


def test_balance_after_halfway_exceeds_half_principal_share():
    # Amortisation front-loads interest, so more than half of a simple linear
    # principal share is still outstanding halfway through the term.
    balance = outstanding_balance(10_000.0, 12.0, 36, 18)
    assert balance > 5_000.0
    assert balance < 6_200.0


def test_monthly_hazard_reproduces_twelve_month_pd():
    hazard = monthly_hazard_from_annual_pd(0.10)
    reconstructed = 1.0 - (1.0 - hazard) ** 12
    assert reconstructed == pytest.approx(0.10, rel=1e-9)


def test_marginal_pd_curve_sums_to_cumulative_pd():
    curve = marginal_pd_curve(np.array([0.08]), horizon_months=24)
    assert curve.shape == (1, 24)
    assert curve.sum() == pytest.approx(cumulative_lifetime_pd(0.08, 24).item(), rel=1e-6)
    # Marginal probabilities must decline as survival shrinks.
    assert curve[0, 0] > curve[0, -1]


def test_discount_factors_are_below_one_and_decreasing():
    factors = discount_factors(0.06, 12)
    assert np.all(factors < 1.0)
    assert np.all(np.diff(factors) < 0)


def test_higher_rate_means_smaller_discount_factor():
    cheaper = discount_factors(0.03, 12)[-1]
    dearer = discount_factors(0.15, 12)[-1]
    assert cheaper > dearer
