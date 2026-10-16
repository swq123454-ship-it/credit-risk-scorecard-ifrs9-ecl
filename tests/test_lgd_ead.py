"""Tests for the LGD and EAD models.

The central assertions are about *discipline*, not accuracy:

* the LGD model must beat the grade-average benchmark out of time, or it is
  complexity without value;
* the LGD feature set must exclude anything only observable after default;
* the EAD model's exposure ratio must fall between zero and one and decline with
  seasoning, because earlier defaults leave more principal outstanding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditrisk import config as cfg
from creditrisk.ead import exposure_profile_table, fit_ead_model, validate_ead_model
from creditrisk.lgd import (
    LGD_FEATURES,
    build_lgd_training_frame,
    fit_lgd_model,
    grade_benchmark_table,
    validate_lgd_model,
)


def _default_book(n: int = 3000, seed: int = 21) -> pd.DataFrame:
    """Synthetic workouts whose loss depends on more than grade.

    If the loss varied only by grade, a grade-average lookup would be unbeatable
    and the LGD model could never justify itself - which is the assumption the
    tests below are meant to challenge.
    """
    rng = np.random.default_rng(seed)
    grades = rng.choice(list("ABCDEFG"), n)
    grade_index = np.array([list("ABCDEFG").index(g) for g in grades])
    loan_amnt = rng.uniform(2_000, 30_000, n).round(2)
    term_months = rng.choice([36, 60], n)
    annual_inc = rng.uniform(20_000, 200_000, n).round(0)
    months_to_default = rng.integers(3, np.maximum(term_months - 2, 5))

    observed_lgd = np.clip(
        0.70
        + 0.030 * grade_index
        + 0.0000045 * loan_amnt
        + 0.0018 * (term_months - 36)
        - 0.0000006 * annual_inc
        + rng.normal(0, 0.030, n),
        0.05,
        0.99,
    )
    out_prncp = (loan_amnt * rng.uniform(0.4, 0.95, n)).round(2)

    return pd.DataFrame(
        {
            "loan_id": [f"L{i}" for i in range(n)],
            "grade": grades,
            "loan_status": cfg.DEFAULT_LABEL,
            "issue_year": rng.choice([2013, 2014, 2015], n),
            "loan_amnt": loan_amnt,
            "int_rate": rng.uniform(6.0, 28.0, n).round(2),
            "term_months": term_months,
            "fico_low": rng.integers(620, 800, n).astype(float),
            "dti": rng.uniform(0, 40, n).round(2),
            "annual_inc": annual_inc,
            "purpose": rng.choice(["debt_consolidation", "credit_card", "car"], n),
            "addr_state": rng.choice(["CA", "TX", "NY"], n),
            "out_prncp": out_prncp,
            "recoveries": (out_prncp * (1 - observed_lgd)).round(2),
            "months_to_default": months_to_default.astype(float),
        }
    )


def _losses(book: pd.DataFrame) -> pd.DataFrame:
    """Attach the derived workout loss the models consume."""
    return build_lgd_training_frame(book, vintages=(2013, 2015))


def test_lgd_feature_set_excludes_post_default_information():
    """Time-to-default is known for a workout, not for a performing loan."""
    assert "months_to_default" not in LGD_FEATURES
    assert "recoveries" not in LGD_FEATURES
    assert "out_prncp" not in LGD_FEATURES
    assert "observed_lgd" not in LGD_FEATURES


def test_build_lgd_training_frame_derives_loss_from_recoveries():
    book = _default_book(n=500)
    frame = build_lgd_training_frame(book, vintages=(2013, 2015))
    assert (frame["observed_lgd"] > 0).all()
    assert (frame["observed_lgd"] < 1).all()
    expected = 1 - frame["recoveries"] / frame["out_prncp"]
    assert frame["observed_lgd"].to_numpy() == pytest.approx(expected.to_numpy(), rel=1e-3)


def test_lgd_model_predictions_respect_floor_and_cap():
    losses = _losses(_default_book())
    model = fit_lgd_model(losses)
    predictions = model.predict_lgd(losses.sample(200, random_state=1))
    assert predictions.min() >= cfg.ECL_ASSUMPTIONS.lgd_floor - 1e-9
    assert predictions.max() <= cfg.ECL_ASSUMPTIONS.lgd_cap + 1e-9


def test_lgd_model_beats_the_grade_average_benchmark_out_of_time():
    train = _losses(_default_book(n=6000, seed=31))
    holdout = _losses(_default_book(n=2000, seed=32))
    model = fit_lgd_model(train)
    _, summary = validate_lgd_model(model, holdout, "holdout")
    assert summary["model_mae"] < summary["benchmark_mae"]
    assert summary["mae_improvement_pct"] > 0


def test_lgd_model_requires_enough_defaults():
    with pytest.raises(RuntimeError):
        fit_lgd_model(_losses(_default_book(n=50)))


def test_grade_benchmark_table_covers_every_grade_present():
    book = _default_book(n=1500)
    table = grade_benchmark_table(_losses(book))
    assert set(table["grade"]) == set(book["grade"].unique())
    assert (table["mean_lgd"].between(0, 1)).all()


def test_ead_model_predicts_exposure_within_principal():
    book = _default_book()
    model = fit_ead_model(_losses(book))
    frame = book.copy()
    frame["months_to_reporting_date"] = 6
    ead, ratio = model.predict_ead(frame)
    assert (ead >= 0).all()
    assert (ead <= frame["loan_amnt"].to_numpy() + 1e-6).all()
    assert (ratio >= 0).all() and (ratio <= 1).all()


def test_earlier_defaults_leave_more_exposure_outstanding():
    book = _default_book()
    model = fit_ead_model(_losses(book))

    early = book.copy()
    early["months_to_reporting_date"] = 2
    late = book.copy()
    late["months_to_reporting_date"] = 30

    _, early_ratio = model.predict_ead(early)
    _, late_ratio = model.predict_ead(late)
    assert early_ratio.mean() > late_ratio.mean()


def test_ead_validation_reports_a_small_mean_difference():
    train = _losses(_default_book(n=6000, seed=41))
    holdout = _losses(_default_book(n=1500, seed=42))
    model = fit_ead_model(train)
    by_grade, summary = validate_ead_model(model, holdout)
    assert summary["defaults"] == 1500
    assert not by_grade.empty
    assert abs(summary["predicted_mean_exposure_ratio"] - summary["observed_mean_exposure_ratio"]) < 0.15


def test_exposure_profile_months_are_ordered():
    book = _default_book(n=4000)
    model = fit_ead_model(_losses(book))
    profile = exposure_profile_table(model)
    assert (profile["p25_default_month"] <= profile["median_default_month"]).all()
    assert (profile["median_default_month"] <= profile["p75_default_month"]).all()


def test_ead_model_without_defaults_raises():
    with pytest.raises(RuntimeError):
        fit_ead_model(pd.DataFrame())
