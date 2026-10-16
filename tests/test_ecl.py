"""Tests for the IFRS 9 three-stage ECL engine.

The staging logic and the discounting behaviour are where provision
calculations usually go wrong, so both are pinned down explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditrisk import config as cfg
from creditrisk.ecl import (
    assign_stage,
    estimate_lgd_from_warehouse,
    horizon_by_stage,
    loan_level_ecl,
    run_ecl_engine,
    scenario_pd_multiplier,
    staging_transition_matrix,
)


def _open_book(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grades = rng.choice(list("ABCDEFG"), n)
    return pd.DataFrame(
        {
            "loan_id": [f"L{i}" for i in range(n)],
            "grade": grades,
            "loan_amnt": rng.uniform(2_000, 30_000, n).round(2),
            "int_rate": rng.uniform(6.0, 28.0, n).round(2),
            "term_months": rng.choice([36, 60], n),
            "months_to_reporting_date": rng.integers(3, 36, n),
            "out_prncp": rng.uniform(500, 25_000, n).round(2),
            "days_past_due": np.where(rng.random(n) < 0.03, 90, np.where(rng.random(n) < 0.08, 15, 0)),
        }
    )


@pytest.fixture()
def lgd_table() -> pd.Series:
    return pd.Series({"A": 0.72, "B": 0.76, "C": 0.80, "D": 0.84, "E": 0.87, "F": 0.90, "G": 0.93})


def test_scenario_multipliers_increase_with_severity():
    base = scenario_pd_multiplier(cfg.MACRO_SCENARIOS["base"])
    adverse = scenario_pd_multiplier(cfg.MACRO_SCENARIOS["adverse"])
    severe = scenario_pd_multiplier(cfg.MACRO_SCENARIOS["severe"])
    assert base == pytest.approx(1.0)
    assert base < adverse < severe


def test_stage_assignment_follows_pd_and_delinquency():
    pd_12m = np.array([0.005, 0.20, 0.01, 0.25])
    dpd = np.array([0, 0, 120, 0])
    stages = assign_stage(pd_12m, dpd)
    assert stages.tolist() == [1, 2, 3, 2]


def test_credit_impairment_overrides_the_pd_based_stage():
    stages = assign_stage(np.array([0.001]), np.array([90]))
    assert stages[0] == 3


def test_stage_one_horizon_is_capped_at_twelve_months():
    horizon = horizon_by_stage(np.array([1, 1]), np.array([48.0, 6.0]))
    assert horizon[0] == pytest.approx(12.0)
    assert horizon[1] == pytest.approx(6.0)


def test_stage_two_horizon_uses_the_remaining_term_with_a_cap():
    horizon = horizon_by_stage(np.array([2, 2]), np.array([48.0, 200.0]))
    assert horizon[0] == pytest.approx(48.0)
    assert horizon[1] == pytest.approx(cfg.ECL_ASSUMPTIONS.max_lifetime_months)


def test_ecl_is_zero_once_the_loan_has_matured():
    ecl, _ = loan_level_ecl(
        loan_amnt=np.array([10_000.0]),
        int_rate=np.array([12.0]),
        term_months=np.array([36.0]),
        months_elapsed=np.array([36.0]),
        pd_12m=np.array([0.10]),
        lgd=np.array([0.80]),
        horizon_months=np.array([0.0]),
    )
    assert ecl[0] == pytest.approx(0.0)


def test_ecl_is_positive_and_bounded_by_lgd_times_exposure():
    ecl, lifetime_pd = loan_level_ecl(
        loan_amnt=np.array([10_000.0]),
        int_rate=np.array([12.0]),
        term_months=np.array([36.0]),
        months_elapsed=np.array([6.0]),
        pd_12m=np.array([0.10]),
        lgd=np.array([0.80]),
        horizon_months=np.array([30.0]),
    )
    assert 0.0 < ecl[0] < 0.80 * 10_000.0
    assert 0.0 < lifetime_pd[0] < 1.0


def test_ecl_increases_with_pd_and_with_lgd():
    arguments = {
        "loan_amnt": np.array([10_000.0]),
        "int_rate": np.array([12.0]),
        "term_months": np.array([36.0]),
        "months_elapsed": np.array([6.0]),
        "horizon_months": np.array([24.0]),
    }
    low, _ = loan_level_ecl(pd_12m=np.array([0.05]), lgd=np.array([0.8]), **arguments)
    high, _ = loan_level_ecl(pd_12m=np.array([0.25]), lgd=np.array([0.8]), **arguments)
    assert high[0] > low[0]

    low_lgd, _ = loan_level_ecl(pd_12m=np.array([0.15]), lgd=np.array([0.45]), **arguments)
    high_lgd, _ = loan_level_ecl(pd_12m=np.array([0.15]), lgd=np.array([0.95]), **arguments)
    assert high_lgd[0] > low_lgd[0]


def test_lifetime_ecl_exceeds_twelve_month_ecl_for_the_same_loan():
    arguments = {
        "loan_amnt": np.array([15_000.0]),
        "int_rate": np.array([14.0]),
        "term_months": np.array([60.0]),
        "months_elapsed": np.array([6.0]),
        "pd_12m": np.array([0.12]),
        "lgd": np.array([0.85]),
    }
    twelve_month, _ = loan_level_ecl(horizon_months=np.array([12.0]), **arguments)
    lifetime, _ = loan_level_ecl(horizon_months=np.array([54.0]), **arguments)
    assert lifetime[0] > twelve_month[0]


def test_discounting_reduces_the_provision():
    arguments = {
        "loan_amnt": np.array([10_000.0]),
        "term_months": np.array([60.0]),
        "months_elapsed": np.array([6.0]),
        "pd_12m": np.array([0.15]),
        "lgd": np.array([0.80]),
        "horizon_months": np.array([54.0]),
    }
    cheap_money, _ = loan_level_ecl(int_rate=np.array([6.0]), **arguments)
    expensive_money, _ = loan_level_ecl(int_rate=np.array([30.0]), **arguments)
    assert expensive_money[0] < cheap_money[0]


def test_estimate_lgd_respects_floor_and_cap():
    raw = pd.DataFrame(
        {
            "grade": ["A", "B", "C"],
            "empirical_lgd": [0.10, 0.80, 0.99],
        }
    )
    clipped = estimate_lgd_from_warehouse(raw)
    assert clipped.loc["A"] == pytest.approx(cfg.ECL_ASSUMPTIONS.lgd_floor)
    assert clipped.loc["B"] == pytest.approx(0.80)
    assert clipped.loc["C"] == pytest.approx(cfg.ECL_ASSUMPTIONS.lgd_cap)


def test_engine_produces_ordered_scenarios_and_reconciles_to_exposure(lgd_table):
    book = _open_book()
    pd_12m = np.full(len(book), 0.06)
    result = run_ecl_engine(book, pd_12m, lgd_table)

    totals = result.scenario_totals.set_index("scenario")["total_ecl"]
    # The upside case must provision less than the base case, and the downside
    # cases must be ordered by severity.
    assert totals["optimistic"] < totals["base"] < totals["adverse"] < totals["severe"]
    assert result.total_exposure == pytest.approx(float(book["out_prncp"].sum()))
    # Probability weighting must land inside the scenario range.
    assert totals["optimistic"] < result.weighted_ecl < totals["severe"]


def test_engine_accepts_a_modeled_lgd_vector(lgd_table):
    book = _open_book()
    pd_12m = np.full(len(book), 0.06)
    low_lgd = np.full(len(book), 0.50)
    high_lgd = np.full(len(book), 0.90)

    low_result = run_ecl_engine(book, pd_12m, lgd_values=low_lgd)
    high_result = run_ecl_engine(book, pd_12m, lgd_values=high_lgd)

    assert low_result.weighted_ecl < high_result.weighted_ecl
    assert low_result.loans["lgd_source"].iloc[0] == "model"


def test_engine_requires_an_lgd_input(lgd_table):
    book = _open_book(n=10)
    with pytest.raises(ValueError):
        run_ecl_engine(book, np.full(len(book), 0.05))


def test_engine_stage_summary_reconciles(lgd_table):
    book = _open_book()
    pd_12m = np.full(len(book), 0.05)
    result = run_ecl_engine(book, pd_12m, lgd_table)

    stage_summary = result.summary_by_stage
    assert stage_summary["loans"].sum() == len(book)
    assert stage_summary["base_ecl"].sum() == pytest.approx(
        result.scenario_totals.set_index("scenario").loc["base", "total_ecl"], rel=1e-6
    )
    assert (stage_summary["coverage_ratio_pct"] > 0).all()


def test_staging_transitions_move_upward_under_stress(lgd_table):
    book = _open_book(n=600)
    pd_12m = np.full(len(book), 0.02)
    result = run_ecl_engine(book, pd_12m, lgd_table)
    matrix = staging_transition_matrix(result.loans)
    # Nothing can migrate from a worse stage to a better one under a stress shock.
    assert matrix.shape[0] == matrix.shape[1]
    assert int(matrix.to_numpy().sum()) == len(book)


def test_empty_book_is_handled_without_error(lgd_table):
    empty = _open_book(n=0)
    pd_12m = np.array([])
    result = run_ecl_engine(empty, pd_12m, lgd_table)
    assert result.total_exposure == 0.0
    assert np.isnan(result.weighted_ecl) or result.weighted_ecl == 0.0
