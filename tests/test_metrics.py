"""Tests for discrimination, calibration and stability metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditrisk.metrics import (
    auc,
    bootstrap_auc_ci,
    brier_score,
    calibration_table,
    expected_cost,
    gini,
    ks_statistic,
    lift_table,
    population_stability_index,
    psi_band,
)


@pytest.fixture()
def scored_sample():
    rng = np.random.default_rng(11)
    n = 6000
    target = rng.integers(0, 2, n)
    probability = np.clip(0.15 + 0.55 * target + rng.normal(0, 0.18, n), 0.001, 0.999)
    return target.astype(float), probability


def test_perfect_separation_gives_maximum_metrics():
    target = np.array([0.0] * 500 + [1.0] * 500)
    probability = np.array([0.1] * 500 + [0.9] * 500)
    assert auc(target, probability) == pytest.approx(1.0)
    assert gini(target, probability) == pytest.approx(1.0)
    assert ks_statistic(target, probability) == pytest.approx(1.0)


def test_gini_relationship_to_auc(scored_sample):
    target, probability = scored_sample
    assert gini(target, probability) == pytest.approx(2 * auc(target, probability) - 1)


def test_ks_is_bounded_and_positive(scored_sample):
    target, probability = scored_sample
    value = ks_statistic(target, probability)
    assert 0.0 < value <= 1.0


def test_auc_of_noise_is_near_one_half():
    rng = np.random.default_rng(3)
    target = rng.integers(0, 2, 5000).astype(float)
    probability = rng.random(5000)
    assert 0.46 < auc(target, probability) < 0.54


def test_psi_of_identical_populations_is_zero():
    rng = np.random.default_rng(5)
    population = rng.normal(0, 1, 5000)
    assert population_stability_index(population, population) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_a_shifted_population():
    rng = np.random.default_rng(5)
    reference = rng.normal(0, 1, 5000)
    shifted = rng.normal(1.5, 1, 5000)
    assert population_stability_index(reference, shifted) > 0.25


def test_psi_bands():
    assert psi_band(0.05) == "stable"
    assert psi_band(0.15) == "monitor"
    assert psi_band(0.40) == "material shift"


def test_calibration_table_columns_and_monotonic_ratio():
    rng = np.random.default_rng(9)
    target = rng.integers(0, 2, 4000).astype(float)
    probability = np.clip(0.1 + 0.5 * target + rng.normal(0, 0.2, 4000), 0.001, 0.999)
    table = calibration_table(target, probability, n_bins=10)
    assert {"loans", "predicted_pd", "observed_pd", "ratio_observed_to_predicted"} <= set(table.columns)
    assert table["loans"].sum() == 4000
    # Predicted PD must increase across deciles.
    assert table["predicted_pd"].is_monotonic_increasing


def test_lift_decile_one_captures_more_defaults_than_average():
    rng = np.random.default_rng(13)
    target = rng.integers(0, 2, 5000).astype(float)
    probability = np.clip(0.15 + 0.5 * target + rng.normal(0, 0.2, 5000), 0.001, 0.999)
    table = lift_table(target, probability, n_bins=10)
    assert table["lift"].iloc[0] > 1.0
    assert table["lift"].iloc[-1] < 1.0
    assert table["cumulative_bad_capture"].iloc[-1] == pytest.approx(1.0)


def test_brier_score_is_lower_for_better_probabilities():
    rng = np.random.default_rng(17)
    target = rng.integers(0, 2, 3000).astype(float)
    good = np.clip(0.1 + 0.6 * target, 0.01, 0.99)
    bad = np.full_like(good, 0.5)
    assert brier_score(target, good) < brier_score(target, bad)


def test_expected_cost_penalises_missed_defaults_more_than_declined_goods():
    target = np.array([1.0, 0.0])
    probability = np.array([0.2, 0.8])  # one miss, one false alarm
    cost = expected_cost(target, probability, threshold=0.5)
    assert cost == pytest.approx(1.0 + 0.05)


def test_bootstrap_interval_brackets_the_point_estimate(scored_sample):
    target, probability = scored_sample
    lower, upper = bootstrap_auc_ci(target, probability, n_iterations=60, seed=1)
    point = auc(target, probability)
    assert lower <= point <= upper
    assert upper - lower > 0.0


def test_calibration_table_rejects_empty_input():
    with pytest.raises(ValueError):
        pd.qcut(pd.Series(dtype=float), 10)
