"""Exposure at default model.

The honest framing for amortising retail term loans
---------------------------------------------------
Unlike a revolving credit line, an amortising loan has no undrawn commitment, so
EAD is not "how much the borrower will draw" - it is "how much principal will
still be outstanding when the borrower defaults". That is a *timing* problem, not
a limit problem.

So the EAD model here is a two-part construction:

1. **Default timing.** Estimate, per grade, the discrete distribution of the
   month in which a loan defaults (a hazard profile), estimated on resolved
   training defaults with Laplace smoothing.
2. **Exposure profile.** Combine that timing distribution with the contractual
   amortisation schedule to get the expected outstanding balance at default.

The output is a single EAD per loan plus an implied exposure ratio
(``EAD / original principal``), which is the retail analogue of a credit
conversion factor.

Validation compares the model's predicted exposure ratio against the realised
ratio on out-of-time defaults (``out_prncp / loan_amnt``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg
from .finmath import outstanding_balance

_GRADE_LETTERS = "ABCDEFG"


@dataclass
class EadModel:
    """Default timing profile per grade, expressed as a probability vector."""

    timing_by_grade: dict[str, np.ndarray]
    max_month: int
    overall_timing: np.ndarray
    assumptions: cfg.EclAssumptions

    def timing_for(self, grade: str) -> np.ndarray:
        return self.timing_by_grade.get(grade, self.overall_timing)

    def predict_ead(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (ead, exposure_ratio) per loan.

        The distribution is renormalised so that default can only occur after the
        loan's current seasoning: a loan that is already 20 months old cannot
        default in month 5.
        """
        frame = frame.reset_index(drop=True)
        loan_amnt = frame["loan_amnt"].to_numpy(dtype=float)
        int_rate = frame["int_rate"].to_numpy(dtype=float)
        term_months = frame["term_months"].to_numpy(dtype=float)
        seasoning = frame["months_to_reporting_date"].to_numpy(dtype=float)
        grades = frame["grade"].astype(str).to_numpy()

        n_loans = len(frame)
        weighted_exposure = np.zeros(n_loans, dtype=float)
        weight_total = np.zeros(n_loans, dtype=float)

        for month in range(1, self.max_month + 1):
            eligible = (month > seasoning) & (month <= term_months)
            if not np.any(eligible):
                continue

            probabilities = np.zeros(n_loans, dtype=float)
            for grade in np.unique(grades[eligible]):
                mask = eligible & (grades == grade)
                probabilities[mask] = self.timing_for(grade)[month - 1]

            exposure = outstanding_balance(loan_amnt, int_rate, term_months, month)
            weighted_exposure += probabilities * exposure
            weight_total += probabilities

        safe_total = np.where(weight_total > 0, weight_total, np.nan)
        ead = np.where(weight_total > 0, weighted_exposure / safe_total, 0.0)
        ead = np.clip(ead, 0.0, loan_amnt)
        exposure_ratio = np.where(loan_amnt > 0, ead / loan_amnt, 0.0)
        return ead, exposure_ratio


def fit_ead_model(
    training_defaults: pd.DataFrame,
    assumptions: cfg.EclAssumptions = cfg.ECL_ASSUMPTIONS,
    smoothing: float = 1.0,
) -> EadModel:
    """Estimate default timing per grade from resolved training defaults."""
    if training_defaults.empty:
        raise RuntimeError("Cannot fit an EAD model without defaulted training observations")

    months = training_defaults["months_to_default"].to_numpy(dtype=float)
    months = np.clip(np.round(months), 1, assumptions.max_lifetime_months).astype(int)
    max_month = int(max(months.max(), 12))

    def profile(values: np.ndarray) -> np.ndarray:
        counts = np.bincount(values, minlength=max_month + 1)[1 : max_month + 1].astype(float)
        counts += smoothing
        return counts / counts.sum()

    timing_by_grade: dict[str, np.ndarray] = {}
    for grade in _GRADE_LETTERS:
        subset = months[training_defaults["grade"].astype(str).to_numpy() == grade]
        if subset.size >= 30:
            timing_by_grade[grade] = profile(subset)

    return EadModel(
        timing_by_grade=timing_by_grade,
        max_month=max_month,
        overall_timing=profile(months),
        assumptions=assumptions,
    )


def validate_ead_model(model: EadModel, holdout_defaults: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare predicted exposure ratios against realised ones."""
    if holdout_defaults.empty:
        return pd.DataFrame(), {}

    frame = holdout_defaults.copy()
    # Seasoning is set to zero so the model produces the UNCONDITIONAL expected
    # exposure: E[balance at default] from origination. The reporting-date book
    # needs the version conditioned on survival to the reporting date (default
    # cannot already have happened); a realised default needs the unconditional
    # expectation, otherwise the prediction is conditioned on information
    # (the default month) that the model would not have had.
    frame["months_to_reporting_date"] = 0
    _, predicted_ratio = model.predict_ead(frame)
    observed_ratio = (frame["out_prncp"] / frame["loan_amnt"]).to_numpy(dtype=float)

    summary = {
        "defaults": int(len(frame)),
        "observed_mean_exposure_ratio": round(float(observed_ratio.mean()), 4),
        "predicted_mean_exposure_ratio": round(float(predicted_ratio.mean()), 4),
        "mae_exposure_ratio": round(float(np.mean(np.abs(observed_ratio - predicted_ratio))), 4),
    }

    table = frame[["grade"]].copy()
    table["observed_ratio"] = observed_ratio
    table["predicted_ratio"] = predicted_ratio
    by_grade = (
        table.groupby("grade")
        .agg(
            defaults=("observed_ratio", "size"),
            observed_ratio=("observed_ratio", "mean"),
            predicted_ratio=("predicted_ratio", "mean"),
        )
        .reset_index()
        .round(4)
    )
    return by_grade, summary


def exposure_profile_table(model: EadModel, grades: list[str] | None = None) -> pd.DataFrame:
    """Cumulative default timing by grade - the shape a credit committee asks for."""
    grades = grades or list(_GRADE_LETTERS)
    rows = []
    for grade in grades:
        timing = model.timing_for(grade)
        rows.append(
            {
                "grade": grade,
                "median_default_month": int(np.searchsorted(np.cumsum(timing), 0.5) + 1),
                "p25_default_month": int(np.searchsorted(np.cumsum(timing), 0.25) + 1),
                "p75_default_month": int(np.searchsorted(np.cumsum(timing), 0.75) + 1),
            }
        )
    return pd.DataFrame(rows)
