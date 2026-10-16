"""Discrimination, calibration and stability metrics used to validate the PD model.

Metric choices are deliberate:

* **Gini / KS** are the reporting convention for credit scorecards. AUC is
  included because it is the vocabulary everyone else speaks.
* **PSI** answers a question AUC cannot: "has the *population* moved?", which is
  what actually triggers model redevelopment in a bank.
* **Calibration** gets first-class treatment because a PD model is a *number*
  fed into a provision calculation, not just a ranking. A model with great Gini
  and broken calibration misstates regulatory capital and provisions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(y_true, probabilities))


def gini(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Gini = 2 * AUC - 1. The standard credit-risk discrimination measure."""
    return float(2.0 * auc(y_true, probabilities) - 1.0)


def ks_statistic(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: maximum separation between the cumulative
    distributions of defaulters and non-defaulters.

    Unlike AUC this reads in the units a credit committee cares about: "the
    score cutoff that captures 60% of defaults rejects x% of good customers".
    """
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(-probabilities)
    y_sorted = y_true[order]

    total_bad = y_sorted.sum()
    total_good = len(y_sorted) - total_bad
    if total_bad == 0 or total_good == 0:
        return 0.0

    cumulative_bad = np.cumsum(y_sorted) / total_bad
    cumulative_good = np.cumsum(1.0 - y_sorted) / total_good
    return float(np.max(np.abs(cumulative_bad - cumulative_good)))


def ks_curve(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(-probabilities)
    y_sorted = y_true[order]
    total_bad = y_sorted.sum()
    total_good = len(y_sorted) - total_bad
    return pd.DataFrame(
        {
            "population_share": np.arange(1, len(y_sorted) + 1) / len(y_sorted),
            "cumulative_bad_share": np.cumsum(y_sorted) / total_bad,
            "cumulative_good_share": np.cumsum(1.0 - y_sorted) / total_good,
        }
    )


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    return float(np.mean((probabilities - y_true) ** 2))


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """PSI between a reference population and a later population.

    Conventional read: ``< 0.10`` stable, ``0.10 - 0.25`` monitor, ``> 0.25``
    material shift / redevelop.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    expected_share = np.histogram(expected, bins=edges)[0] / max(len(expected), 1)
    actual_share = np.histogram(actual, bins=edges)[0] / max(len(actual), 1)

    expected_share = np.clip(expected_share, epsilon, None)
    actual_share = np.clip(actual_share, epsilon, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def calibration_table(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate by score decile."""
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(probabilities, dtype=float)})
    frame["bin"] = pd.qcut(frame["p"], n_bins, duplicates="drop")
    table = (
        frame.groupby("bin", observed=True)
        .agg(loans=("y", "size"), predicted_pd=("p", "mean"), observed_pd=("y", "mean"))
        .reset_index(drop=True)
    )
    table["ratio_observed_to_predicted"] = table["observed_pd"] / table["predicted_pd"].replace(0, np.nan)
    return table


def lift_table(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(probabilities, dtype=float)})
    frame = frame.sort_values("p", ascending=False).reset_index(drop=True)
    frame["decile"] = pd.qcut(frame.index, n_bins, labels=range(1, n_bins + 1))
    overall_bad_rate = frame["y"].mean()
    table = (
        frame.groupby("decile", observed=True)
        .agg(loans=("y", "size"), defaults=("y", "sum"), bad_rate=("y", "mean"))
        .reset_index()
    )
    table["lift"] = table["bad_rate"] / overall_bad_rate
    table["cumulative_bad_capture"] = table["defaults"].cumsum() / frame["y"].sum()
    return table


def expected_cost(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    cost_false_negative: float = 1.0,
    cost_false_positive: float = 0.05,
) -> float:
    """Total misclassification cost at a decision threshold.

    ``cost_false_negative`` is the loss from approving a loan that defaults;
    ``cost_false_positive`` is the profit foregone by declining a good applicant.
    Expressing costs as multiples of the same unit is what turns a model score
    into an actual cut-off policy.
    """
    y_true = np.asarray(y_true, dtype=float)
    predictions = (np.asarray(probabilities, dtype=float) >= threshold).astype(float)
    false_negatives = np.sum((predictions == 0) & (y_true == 1))
    false_positives = np.sum((predictions == 1) & (y_true == 0))
    return float(false_negatives * cost_false_negative + false_positives * cost_false_positive)


def bootstrap_auc_ci(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_iterations: int = 200,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for AUC.

    A point estimate of AUC without an interval invites over-reading small
    differences between models that are not statistically distinguishable.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    n = len(y_true)
    estimates = []
    for _ in range(n_iterations):
        index = rng.integers(0, n, n)
        sample_y = y_true[index]
        if sample_y.sum() == 0 or sample_y.sum() == n:
            continue
        estimates.append(roc_auc_score(sample_y, probabilities[index]))
    if not estimates:
        return (float("nan"), float("nan"))
    lower = float(np.quantile(estimates, (1 - confidence) / 2))
    upper = float(np.quantile(estimates, 1 - (1 - confidence) / 2))
    return lower, upper


def roc_frame(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    return pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr})


def decile_monotonicity(table: pd.DataFrame, column: str = "bad_rate") -> bool:
    values = table[column].to_numpy(dtype=float)
    return bool((np.diff(values) <= 1e-12).all())


def psi_band(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "monitor"
    return "material shift"
