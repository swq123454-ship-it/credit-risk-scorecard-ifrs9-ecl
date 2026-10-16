"""Logistic-regression scorecard with points scaling.

Implements the standard industry parameterisation:

    score = offset - factor * logit(p)
    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)

with ``base_score = 600`` at odds of ``20:1`` and 50 points to double the odds.
The resulting score sheet is what a credit analyst actually uses, and its
per-bin points are what gets shown to a regulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config as cfg
from .features import WoeEncoder


@dataclass
class Scorecard:
    model: LogisticRegression
    feature_names: list[str]
    factor: float
    offset: float
    base_score: int
    base_odds: int
    pdo: int

    def logit(self, woe_frame: pd.DataFrame) -> np.ndarray:
        matrix = woe_frame[self.feature_names].to_numpy(dtype=float)
        return self.model.intercept_[0] + matrix @ self.model.coef_[0]

    def score(self, woe_frame: pd.DataFrame) -> np.ndarray:
        """Credit score in points. Higher score = lower risk."""
        return self.offset - self.factor * self.logit(woe_frame)

    def predicted_pd(self, woe_frame: pd.DataFrame) -> np.ndarray:
        """Probability of default implied by the score."""
        return 1.0 / (1.0 + np.exp(-self.logit(woe_frame)))


def fit_scorecard(
    woe_train: pd.DataFrame,
    target: pd.Series,
    base_score: int = cfg.SCORECARD_BASE_SCORE,
    base_odds: int = cfg.SCORECARD_BASE_ODDS,
    pdo: int = cfg.SCORECARD_PDO,
    regularisation: float = 1.0,
) -> Scorecard:
    feature_names = list(woe_train.columns)
    model = LogisticRegression(
        C=regularisation,
        solver="lbfgs",
        max_iter=2000,
        random_state=cfg.RANDOM_SEED,
    )
    model.fit(woe_train[feature_names].to_numpy(dtype=float), target.to_numpy(dtype=float))

    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)

    return Scorecard(
        model=model,
        feature_names=feature_names,
        factor=float(factor),
        offset=float(offset),
        base_score=base_score,
        base_odds=base_odds,
        pdo=pdo,
    )


def coefficient_table(scorecard: Scorecard) -> pd.DataFrame:
    coefficients = scorecard.model.coef_[0]
    return (
        pd.DataFrame({"feature": scorecard.feature_names, "coefficient": coefficients})
        .assign(
            odds_multiplier_per_unit_woe=lambda d: np.exp(d["coefficient"]),
            points_per_unit_woe=lambda d: -d["coefficient"] * scorecard.factor,
        )
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )


def scorecard_table(encoder: WoeEncoder, scorecard: Scorecard) -> pd.DataFrame:
    """Expand the model into a human-readable points sheet.

    For every feature and every bin: the bin definition, its WOE, and the points
    contributed to the final score. Total points across all features plus the
    offset reproduces the score exactly.
    """
    # strict: a coefficient row that does not match the feature count is a model
    # wiring bug, and silently zipping would mislabel every bin's points.
    coefficient_by_feature = dict(zip(scorecard.feature_names, scorecard.model.coef_[0], strict=True))
    rows: list[dict] = []

    for feature, scheme in encoder.schemes_.items():
        if feature not in coefficient_by_feature:
            continue
        coefficient = coefficient_by_feature[feature]

        if scheme.kind == "numeric":
            for entry in scheme.numeric_bins:
                lower, upper = entry["lower"], entry["upper"]
                if np.isneginf(lower):
                    label = f"<= {upper:,.2f}"
                elif np.isposinf(upper):
                    label = f"> {lower:,.2f}"
                else:
                    label = f"({lower:,.2f}, {upper:,.2f}]"
                rows.append(
                    {
                        "feature": feature,
                        "bin": label,
                        "loans_in_bin": entry["count"],
                        "bad_rate": round(entry["bad_rate"], 4),
                        "woe": round(entry["woe"], 4),
                        "points": round(-coefficient * entry["woe"] * scorecard.factor, 2),
                    }
                )
            if scheme.missing_count:
                rows.append(
                    {
                        "feature": feature,
                        "bin": "MISSING",
                        "loans_in_bin": scheme.missing_count,
                        "bad_rate": np.nan,
                        "woe": round(scheme.missing_woe, 4),
                        "points": round(-coefficient * scheme.missing_woe * scorecard.factor, 2),
                    }
                )
        else:
            for category, woe in scheme.category_woe.items():
                rows.append(
                    {
                        "feature": feature,
                        "bin": category,
                        "loans_in_bin": scheme.category_counts.get(category, 0),
                        "bad_rate": np.nan,
                        "woe": round(woe, 4),
                        "points": round(-coefficient * woe * scorecard.factor, 2),
                    }
                )

    return pd.DataFrame(rows)


def score_to_rating_band(scores: np.ndarray, n_bands: int = 8) -> np.ndarray:
    """Map scores onto internal rating grades (1 = best) on equal-frequency bands."""
    ranks = pd.Series(scores).rank(method="first").to_numpy()
    band = np.minimum((ranks / len(ranks) * n_bands).astype(int), n_bands - 1) + 1
    return band
