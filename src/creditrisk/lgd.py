"""Loss given default model.

Why not just use a grade-average LGD?
-------------------------------------
Because it is a constant. A flat LGD by grade assigns the same loss to a
EUR 3,000 loan and a EUR 30,000 loan with identical grade and term, which is
empirically wrong and materially misstates a provision on a skewed book. The
model here regresses the observed workout loss on origination characteristics
and is only kept if it beats the grade-average benchmark on the out-of-time
default sample. That comparison is the point: an LGD model that cannot beat a
lookup table is complexity without value.

Critical modelling constraint
-----------------------------
Features must be observable at the *reporting date* for a loan that has not yet
defaulted. Time-to-default is therefore excluded - it is known for a workout
that already happened, and is unavailable for a performing loan. Including it
would produce an excellent in-sample fit and an unusable model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from . import config as cfg

#: Origination characteristics used to explain the workout loss.
LGD_FEATURES: list[str] = [
    "loan_amnt",
    "int_rate",
    "term_months",
    "fico_low",
    "dti",
    "annual_inc",
    "loan_to_income",
    "grade_code",
    "purpose_code",
    "addr_state_code",
]

_GRADE_CODES: dict[str, int] = {letter: index for index, letter in enumerate("ABCDEFG")}


@dataclass
class LgdModel:
    estimator: HistGradientBoostingRegressor
    feature_names: list[str]
    floor: float
    cap: float
    benchmark_by_grade: pd.Series

    def predict_lgd(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self._prepare(frame)[self.feature_names].to_numpy(dtype=float)
        predictions = self.estimator.predict(matrix)
        return np.clip(predictions, self.floor, self.cap)

    def benchmark_lgd(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame["grade"].map(self.benchmark_by_grade)
        fallback = float(self.benchmark_by_grade.mean())
        return values.fillna(fallback).clip(self.floor, self.cap).to_numpy(dtype=float)

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        prepared["grade_code"] = prepared["grade"].map(_GRADE_CODES).fillna(3.0)
        prepared["purpose_code"] = prepared["purpose"].astype("category").cat.codes.astype(float)
        prepared["addr_state_code"] = prepared["addr_state"].astype("category").cat.codes.astype(float)
        prepared["loan_to_income"] = prepared["loan_amnt"] / prepared["annual_inc"].replace(0, np.nan)
        prepared["loan_to_income"] = prepared["loan_to_income"].fillna(prepared["loan_to_income"].median())
        return prepared


def build_lgd_training_frame(
    mart: pd.DataFrame,
    vintages: tuple[int, int] = cfg.TRAIN_YEARS,
) -> pd.DataFrame:
    """Observed workout loss on resolved defaults.

    ``LGD = 1 - recoveries / exposure_at_default``, where exposure at default is
    the outstanding principal when the loan charged off.
    """
    defaults = mart[mart["loan_status"] == cfg.DEFAULT_LABEL].copy()
    defaults = defaults[(defaults["issue_year"] >= vintages[0]) & (defaults["issue_year"] <= vintages[1])]
    defaults = defaults[defaults["out_prncp"] > 100]
    defaults["observed_lgd"] = (1.0 - defaults["recoveries"] / defaults["out_prncp"]).clip(0.05, 0.99)
    return defaults.reset_index(drop=True)


def fit_lgd_model(
    training_defaults: pd.DataFrame,
    floor: float = cfg.ECL_ASSUMPTIONS.lgd_floor,
    cap: float = cfg.ECL_ASSUMPTIONS.lgd_cap,
) -> LgdModel:
    if len(training_defaults) < 200:
        raise RuntimeError(
            f"Only {len(training_defaults)} defaulted training loans available; not enough to fit a stable LGD model."
        )

    placeholder = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=cfg.RANDOM_SEED,
    )
    model = LgdModel(
        estimator=placeholder,
        feature_names=LGD_FEATURES,
        floor=floor,
        cap=cap,
        benchmark_by_grade=pd.Series(dtype=float),
    )

    prepared = model._prepare(training_defaults)
    estimator = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=cfg.RANDOM_SEED,
    )
    estimator.fit(prepared[LGD_FEATURES].to_numpy(dtype=float), prepared["observed_lgd"].to_numpy(dtype=float))

    model.estimator = estimator
    model.benchmark_by_grade = training_defaults.groupby("grade")["observed_lgd"].mean()
    return model


def validate_lgd_model(
    model: LgdModel,
    holdout_defaults: pd.DataFrame,
    sample_label: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare the model against the grade-average benchmark on out-of-time data."""
    if holdout_defaults.empty:
        return pd.DataFrame(), {}

    observed = holdout_defaults["observed_lgd"].to_numpy(dtype=float)
    predicted = model.predict_lgd(holdout_defaults)
    benchmark = model.benchmark_lgd(holdout_defaults)

    summary = {
        "sample": sample_label,
        "defaults": int(len(holdout_defaults)),
        "model_mae": round(float(mean_absolute_error(observed, predicted)), 4),
        "benchmark_mae": round(float(mean_absolute_error(observed, benchmark)), 4),
        "model_r2": round(float(r2_score(observed, predicted)), 4),
        "benchmark_r2": round(float(r2_score(observed, benchmark)), 4),
        "observed_mean_lgd": round(float(observed.mean()), 4),
        "model_mean_lgd": round(float(predicted.mean()), 4),
        "benchmark_mean_lgd": round(float(benchmark.mean()), 4),
    }
    summary["mae_improvement_pct"] = round(
        100.0 * (summary["benchmark_mae"] - summary["model_mae"]) / max(summary["benchmark_mae"], 1e-9), 2
    )

    deciles = pd.DataFrame(
        {
            "observed_lgd": observed,
            "model_lgd": predicted,
            "grade": holdout_defaults["grade"].to_numpy(),
        }
    )
    deciles["band"] = pd.qcut(deciles["model_lgd"], 10, duplicates="drop", labels=False)
    table = (
        deciles.groupby("band", observed=True)
        .agg(
            defaults=("observed_lgd", "size"),
            predicted_lgd=("model_lgd", "mean"),
            observed_lgd=("observed_lgd", "mean"),
        )
        .reset_index()
        .round(4)
    )
    return table, summary


def grade_benchmark_table(defaults: pd.DataFrame) -> pd.DataFrame:
    """The incumbent: grade-average LGD, which the model must beat."""
    table = (
        defaults.groupby("grade")
        .agg(
            defaults=("observed_lgd", "size"),
            mean_lgd=("observed_lgd", "mean"),
            median_lgd=("observed_lgd", "median"),
            p10_lgd=("observed_lgd", lambda s: float(np.quantile(s, 0.10))),
            p90_lgd=("observed_lgd", lambda s: float(np.quantile(s, 0.90))),
        )
        .reset_index()
        .round(4)
    )
    return table
