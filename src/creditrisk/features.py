"""Weight-of-evidence binning and information value.

Why WOE at all in 2025, when gradient boosting exists?
------------------------------------------------------
Supervisory expectations. A PD model that a bank cannot explain to its
regulator is not deployable, and a WOE-transformed logistic regression remains
the industry default for *credit decisioning* because:

* each input is reduced to a monotonic, human-readable bin set;
* the model output decomposes into points a credit analyst can read off a sheet;
* stability (PSI) is measurable at the bin level.

The GBM in this project exists to quantify how much discrimination the
interpretable model gives up. It is not a replacement.

Binning algorithm
-----------------
1. Initial equal-frequency bins (quantiles), with missing values held as their
   own bin.
2. Merge adjacent bins until the WOE sequence is monotonic in either direction,
   always merging the adjacent pair whose WOE values are closest, which is the
   pair that loses the least discriminatory information.
3. `IV > 0.5` is reported as *suspicious* rather than good: in credit data that
   usually means a feature has leaked the target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_EPSILON = 1e-6

#: Rule-of-thumb interpretation bands for information value.
IV_BANDS: list[tuple[float, str]] = [
    (0.02, "unpredictive"),
    (0.10, "weak"),
    (0.30, "medium"),
    (0.50, "strong"),
    (float("inf"), "suspicious (check for leakage)"),
]


def interpret_iv(iv: float) -> str:
    for threshold, label in IV_BANDS:
        if iv < threshold:
            return label
    return "suspicious (check for leakage)"


def _woe_for_counts(bad: float, good: float, total_bad: float, total_good: float) -> float:
    bad_share = max(bad, _EPSILON) / max(total_bad, _EPSILON)
    good_share = max(good, _EPSILON) / max(total_good, _EPSILON)
    return math.log(bad_share / good_share)


def _iv_for_bins(bad: np.ndarray, good: np.ndarray) -> float:
    total_bad = bad.sum()
    total_good = good.sum()
    if total_bad <= 0 or total_good <= 0:
        return 0.0
    bad_share = bad / total_bad
    good_share = good / total_good
    woe = np.log((bad_share + _EPSILON) / (good_share + _EPSILON))
    return float(np.sum((bad_share - good_share) * woe))


@dataclass
class BinScheme:
    """A fitted binning rule for one feature."""

    feature: str
    kind: str  # "numeric" or "categorical"
    iv: float
    monotonic: bool
    #: Numeric bins: ordered list of (lower, upper, woe, count, bad_rate)
    numeric_bins: list[dict] = field(default_factory=list)
    missing_woe: float = 0.0
    missing_count: int = 0
    #: Categorical: value -> woe (plus "OTHER" and "__UNSEEN__" keys)
    category_woe: dict[str, float] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)
    unseen_woe: float = 0.0

    def transform_series(self, series: pd.Series) -> pd.Series:
        if self.kind == "categorical":
            mapping = dict(self.category_woe)
            fallback = mapping.get("OTHER", self.unseen_woe)
            return series.astype(object).map(lambda v: mapping.get(str(v), fallback)).astype(float)

        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        output = np.full(values.shape, self.missing_woe, dtype=float)
        for entry in self.numeric_bins:
            lower = entry["lower"]
            upper = entry["upper"]
            mask = (values > lower) & (values <= upper)
            mask &= ~np.isnan(values)
            output[mask] = entry["woe"]
        return pd.Series(output, index=series.index)


def _initial_numeric_bins(values: np.ndarray, max_bins: int) -> np.ndarray:
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return np.array([-np.inf, np.inf])
    quantiles = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
    edges = np.unique(np.quantile(finite, quantiles))
    return np.concatenate([[-np.inf], edges, [np.inf]])


def _merge_to_monotonic(bins: list[dict], total_bad: float, total_good: float) -> list[dict]:
    """Merge adjacent bins until WOE is monotonic (increasing or decreasing).

    WOE is recomputed after every merge: merging two bins changes the evidence
    each bin carries, so a stale WOE would let the loop test a sequence that no
    longer exists.
    """

    def with_woe(entries: list[dict]) -> list[dict]:
        for entry in entries:
            entry["woe"] = _woe_for_counts(entry["bad"], entry["good"], total_bad, total_good)
        return entries

    def is_monotonic(entries: list[dict]) -> bool:
        woe = np.array([b["woe"] for b in entries], dtype=float)
        diffs = np.diff(woe)
        return bool((diffs >= -1e-9).all() or (diffs <= 1e-9).all())

    working = with_woe([dict(b) for b in bins])
    while len(working) > 2 and not is_monotonic(working):
        woe = np.array([b["woe"] for b in working], dtype=float)
        index = int(np.argmin(np.abs(np.diff(woe))))
        left, right = working[index], working[index + 1]
        merged = {
            "lower": left["lower"],
            "upper": right["upper"],
            "count": left["count"] + right["count"],
            "bad": left["bad"] + right["bad"],
            "good": left["good"] + right["good"],
        }
        working[index : index + 2] = [merged]
        working = with_woe(working)
    return working


def fit_numeric_scheme(
    feature: str,
    values: pd.Series,
    target: pd.Series,
    max_bins: int = 10,
    min_bin_pct: float = 0.05,
) -> BinScheme:
    numeric = pd.to_numeric(values, errors="coerce")
    y = target.astype(float)

    total_bad = float(y.sum())
    total_good = float(len(y) - total_bad)

    missing_mask = numeric.isna()
    missing_count = int(missing_mask.sum())
    missing_woe = _woe_for_counts(
        float(y[missing_mask].sum()), float((1 - y[missing_mask]).sum()), total_bad, total_good
    )

    edges = _initial_numeric_bins(numeric.to_numpy(dtype=float), max_bins)
    bin_index = np.digitize(numeric.fillna(edges[0]).to_numpy(dtype=float), edges[1:-1], right=True)
    bin_index = np.where(missing_mask.to_numpy(), -1, bin_index)

    bins: list[dict] = []
    for position in range(len(edges) - 1):
        mask = bin_index == position
        count = int(mask.sum())
        if count == 0:
            continue
        bad = int(y[mask].sum())
        bins.append(
            {
                "lower": float(edges[position]),
                "upper": float(edges[position + 1]),
                "count": count,
                "bad": bad,
                "good": count - bad,
            }
        )

    # Drop bins that are too thin, folding them into the neighbouring bin.
    minimum = max(int(min_bin_pct * len(y)), 1)
    merged_thin: list[dict] = []
    for entry in bins:
        if merged_thin and entry["count"] < minimum:
            previous = merged_thin[-1]
            previous["upper"] = entry["upper"]
            previous["count"] += entry["count"]
            previous["bad"] += entry["bad"]
            previous["good"] += entry["good"]
        else:
            merged_thin.append(dict(entry))

    for entry in merged_thin:
        entry["woe"] = _woe_for_counts(entry["bad"], entry["good"], total_bad, total_good)

    merged = _merge_to_monotonic(merged_thin, total_bad, total_good)
    for entry in merged:
        entry["woe"] = _woe_for_counts(entry["bad"], entry["good"], total_bad, total_good)
        entry["bad_rate"] = entry["bad"] / entry["count"] if entry["count"] else 0.0

    bad_array = np.array([b["bad"] for b in merged], dtype=float)
    good_array = np.array([b["good"] for b in merged], dtype=float)
    iv = _iv_for_bins(bad_array, good_array)

    diffs = np.diff([b["woe"] for b in merged])
    monotonic = bool(len(diffs) == 0 or (diffs >= -1e-9).all() or (diffs <= 1e-9).all())

    return BinScheme(
        feature=feature,
        kind="numeric",
        iv=iv,
        monotonic=monotonic,
        numeric_bins=merged,
        missing_woe=missing_woe,
        missing_count=missing_count,
    )


def fit_categorical_scheme(
    feature: str,
    values: pd.Series,
    target: pd.Series,
    min_bin_pct: float = 0.02,
    max_categories: int = 30,
) -> BinScheme:
    frame = pd.DataFrame({"value": values.astype(object).fillna("__MISSING__"), "target": target.astype(float)})
    grouped = frame.groupby("value")["target"].agg(["count", "sum"])
    grouped = grouped.rename(columns={"sum": "bad"})
    grouped["good"] = grouped["count"] - grouped["bad"]

    total_bad = float(target.sum())
    total_good = float(len(target) - total_bad)

    minimum = max(int(min_bin_pct * len(target)), 1)
    frequent = grouped[grouped["count"] >= minimum].sort_values("count", ascending=False)
    rare = grouped[grouped["count"] < minimum]

    if len(frequent) > max_categories:
        keep = frequent.head(max_categories)
        rare = pd.concat([rare, frequent.iloc[max_categories:]])
        frequent = keep

    category_woe: dict[str, float] = {}
    category_counts: dict[str, int] = {}
    for value, row in frequent.iterrows():
        category_woe[str(value)] = _woe_for_counts(float(row["bad"]), float(row["good"]), total_bad, total_good)
        category_counts[str(value)] = int(row["count"])

    if len(rare) > 0:
        category_woe["OTHER"] = _woe_for_counts(
            float(rare["bad"].sum()), float(rare["good"].sum()), total_bad, total_good
        )
        category_counts["OTHER"] = int(rare["count"].sum())

    unseen_woe = 0.0  # neutral: an unseen category carries no evidence

    bad_array = np.array(
        [grouped.loc[k, "bad"] for k in frequent.index] + ([rare["bad"].sum()] if len(rare) else []), dtype=float
    )
    good_array = np.array(
        [grouped.loc[k, "good"] for k in frequent.index] + ([rare["good"].sum()] if len(rare) else []), dtype=float
    )

    return BinScheme(
        feature=feature,
        kind="categorical",
        iv=_iv_for_bins(bad_array, good_array),
        monotonic=True,
        category_woe=category_woe,
        category_counts=category_counts,
        unseen_woe=unseen_woe,
    )


class WoeEncoder:
    """Fit WOE binning on the training window and apply it everywhere else."""

    def __init__(self, numeric_features: list[str], categorical_features: list[str], max_bins: int = 10):
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.max_bins = max_bins
        self.schemes_: dict[str, BinScheme] = {}
        self.feature_names_: list[str] = list(self.numeric_features) + list(self.categorical_features)

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> WoeEncoder:
        for feature in self.numeric_features:
            if feature not in frame.columns:
                raise KeyError(f"Missing numeric feature in training frame: {feature}")
            self.schemes_[feature] = fit_numeric_scheme(feature, frame[feature], target, max_bins=self.max_bins)
        for feature in self.categorical_features:
            if feature not in frame.columns:
                raise KeyError(f"Missing categorical feature in training frame: {feature}")
            self.schemes_[feature] = fit_categorical_scheme(feature, frame[feature], target)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.schemes_:
            raise RuntimeError("WoeEncoder.transform called before fit()")
        return pd.DataFrame(
            {feature: scheme.transform_series(frame[feature]) for feature, scheme in self.schemes_.items()},
            index=frame.index,
        )

    def iv_table(self) -> pd.DataFrame:
        rows = [
            {
                "feature": feature,
                "information_value": round(scheme.iv, 4),
                "interpretation": interpret_iv(scheme.iv),
                "bins": len(scheme.numeric_bins) if scheme.kind == "numeric" else len(scheme.category_woe),
                "monotonic": scheme.monotonic,
            }
            for feature, scheme in self.schemes_.items()
        ]
        return pd.DataFrame(rows).sort_values("information_value", ascending=False).reset_index(drop=True)

    def selected_features(self, iv_threshold: float = 0.02) -> list[str]:
        """Retain features with a usable level of predictive power.

        Only the *lower* bound is applied automatically. A very high IV is a
        reason to investigate a feature, not a reason to delete it: in this book
        the internal grade and the contract rate both carry an IV around 0.9 and
        both are legitimately predictive - they are the lender's own risk
        assessment. Silently dropping them on a leakage heuristic would throw
        away the two strongest inputs in the model.

        Use :meth:`flagged_features` to surface the features that need a human
        to confirm they are origination-time information before they are used.
        """
        return [feature for feature, scheme in self.schemes_.items() if scheme.iv >= iv_threshold]

    def flagged_features(self, iv_upper: float = 0.5) -> list[str]:
        """Features whose IV is high enough to warrant a leakage review."""
        return [feature for feature, scheme in self.schemes_.items() if scheme.iv >= iv_upper]
