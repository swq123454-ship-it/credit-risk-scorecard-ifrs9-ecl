"""Gradient-boosting challenger model.

Purpose: quantify the discrimination the interpretable scorecard gives up.

Encoding discipline
-------------------
Categorical features are mapped through a **frozen** category-to-code dictionary
built at fit time. Re-deriving codes from the scoring frame is a classic silent
bug: ``pandas`` assigns codes by sort order within the frame, so a category that
is present in training and absent at scoring time shifts every subsequent code,
and the model quietly degrades without raising an error. Unseen categories map to
``-1``.

The implementation prefers LightGBM and falls back to scikit-learn's
``HistGradientBoostingClassifier`` so the pipeline stays runnable where the
LightGBM wheel is unavailable (this matters for CI).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg

try:  # pragma: no cover - environment dependent
    import lightgbm as lgb

    _HAS_LIGHTGBM = True
except Exception:  # pragma: no cover
    _HAS_LIGHTGBM = False

from sklearn.ensemble import HistGradientBoostingClassifier

#: Code assigned to a category that was not present during training.
UNSEEN_CODE = -1.0


def build_category_maps(
    frame: pd.DataFrame,
    categorical_features: list[str],
) -> dict[str, dict[str, float]]:
    """Freeze a category -> code mapping from the training frame."""
    maps: dict[str, dict[str, float]] = {}
    for feature in categorical_features:
        if feature not in frame.columns:
            continue
        values = frame[feature].astype(str)
        counts = values.value_counts()
        maps[feature] = {category: float(code) for code, category in enumerate(counts.index)}
    return maps


def encode_frame(frame: pd.DataFrame, category_maps: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Apply frozen category maps. Unseen categories become -1."""
    encoded = frame.copy()
    for feature, mapping in category_maps.items():
        encoded[feature] = encoded[feature].astype(str).map(mapping).fillna(UNSEEN_CODE).astype(float)
    return encoded


@dataclass
class GbmModel:
    estimator: object
    backend: str
    feature_names: list[str]
    category_maps: dict[str, dict[str, float]] = field(default_factory=dict)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self._matrix(frame)
        return self.estimator.predict_proba(matrix)[:, 1]

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        encoded = encode_frame(frame, self.category_maps)
        return encoded[self.feature_names].to_numpy(dtype=float)


def fit_gbm(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_features: list[str] | None = None,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 5,
) -> GbmModel:
    categorical_features = categorical_features or []
    category_maps = build_category_maps(features, categorical_features)
    matrix = encode_frame(features, category_maps)
    feature_names = list(matrix.columns)
    y = target.to_numpy(dtype=float)

    if _HAS_LIGHTGBM:
        estimator = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            min_child_samples=50,
            reg_lambda=1.0,
            random_state=cfg.RANDOM_SEED,
            verbose=-1,
        )
        backend = "lightgbm"
    else:
        estimator = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=cfg.RANDOM_SEED,
        )
        backend = "sklearn.HistGradientBoosting"

    estimator.fit(matrix[feature_names].to_numpy(dtype=float), y)

    return GbmModel(
        estimator=estimator,
        backend=backend,
        feature_names=feature_names,
        category_maps=category_maps,
    )


def gbm_feature_importance(model: GbmModel) -> pd.DataFrame:
    estimator = model.estimator
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    else:  # pragma: no cover
        importance = np.zeros(len(model.feature_names))
    total = importance.sum() or 1.0
    return (
        pd.DataFrame({"feature": model.feature_names, "importance": importance / total})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
