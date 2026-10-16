"""Figure generation.

Every figure in the README is produced by this module, so no chart can drift
out of sync with the numbers in the report. Figures are written to
``outputs/figures/`` as PNGs with a light background for readability on GitHub.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg

plt.rcParams.update(cfg.PLOT_STYLE)

PALETTE = cfg.PALETTE


def _save(figure, name: str) -> Path:
    path = cfg.FIGURE_DIR / name
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_roc(roc_frames: dict[str, pd.DataFrame], aucs: dict[str, float], name: str = "roc_curve.png") -> Path:
    figure, axes = plt.subplots(figsize=(5.2, 4.6))
    colours = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]
    # The palette holds one colour per model the pipeline can produce; a caller
    # passing more models than colours is a bug, so that direction is strict. The
    # slice covers the legitimate case of fewer models than colours (the scorecard
    # and the challenger are two curves on a three-colour palette).
    if len(roc_frames) > len(colours):
        raise ValueError(f"{len(roc_frames)} models for a {len(colours)}-colour palette")
    for (label, frame), colour in zip(roc_frames.items(), colours, strict=False):
        axes.plot(
            frame["false_positive_rate"],
            frame["true_positive_rate"],
            color=colour,
            linewidth=1.6,
            label=f"{label} (AUC {aucs[label]:.3f})",
        )
    axes.plot([0, 1], [0, 1], color=PALETTE["muted"], linewidth=1.0, linestyle="--", label="Random")
    axes.set_xlabel("False positive rate (good customers declined)")
    axes.set_ylabel("True positive rate (defaults captured)")
    axes.set_title("ROC curve - out-of-time sample (2017)")
    axes.legend(loc="lower right", frameon=False, fontsize=8)
    return _save(figure, name)


def plot_ks(ks_frame: pd.DataFrame, ks_value: float, name: str = "ks_plot.png") -> Path:
    figure, axes = plt.subplots(figsize=(5.6, 4.4))
    axes.plot(
        ks_frame["population_share"],
        ks_frame["cumulative_bad_share"],
        color=PALETTE["danger"],
        linewidth=1.6,
        label="Cumulative defaulters",
    )
    axes.plot(
        ks_frame["population_share"],
        ks_frame["cumulative_good_share"],
        color=PALETTE["primary"],
        linewidth=1.6,
        label="Cumulative non-defaulters",
    )
    gap = np.abs(ks_frame["cumulative_bad_share"] - ks_frame["cumulative_good_share"])
    index = int(gap.idxmax())
    axes.annotate(
        f"KS = {ks_value:.3f}",
        xy=(ks_frame["population_share"].iloc[index], ks_frame["cumulative_bad_share"].iloc[index]),
        xytext=(ks_frame["population_share"].iloc[index] + 0.12, 0.42),
        arrowprops={"arrowstyle": "->", "color": PALETTE["muted"], "linewidth": 1.0},
        fontsize=9,
    )
    axes.set_xlabel("Population ordered by predicted risk (highest risk first)")
    axes.set_ylabel("Cumulative share")
    axes.set_title("Kolmogorov-Smirnov separation - out-of-time sample")
    axes.legend(loc="lower right", frameon=False, fontsize=8)
    return _save(figure, name)


def plot_calibration(calibration_frames: dict[str, pd.DataFrame], name: str = "calibration.png") -> Path:
    figure, axes = plt.subplots(figsize=(5.2, 4.4))
    colours = {
        "Uncalibrated scorecard": PALETTE["secondary"],
        "Calibrated scorecard": PALETTE["primary"],
        "GBM challenger": PALETTE["accent"],
    }
    maximum = 0.0
    for label, frame in calibration_frames.items():
        axes.plot(
            frame["predicted_pd"],
            frame["observed_pd"],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            color=colours.get(label, PALETTE["muted"]),
            label=label,
        )
        maximum = max(maximum, float(frame["predicted_pd"].max()), float(frame["observed_pd"].max()))
    limit = max(maximum * 1.15, 0.1)
    axes.plot(
        [0, limit], [0, limit], color=PALETTE["muted"], linestyle="--", linewidth=1.0, label="Perfect calibration"
    )
    axes.set_xlabel("Predicted PD")
    axes.set_ylabel("Observed default rate")
    axes.set_title("Calibration - out-of-time sample")
    axes.legend(loc="upper left", frameon=False, fontsize=8)
    return _save(figure, name)


def plot_psi_heatmap(psi_frame: pd.DataFrame, name: str = "psi_heatmap.png") -> Path:
    columns = [c for c in psi_frame.columns if c != "feature"]
    values = psi_frame[columns].to_numpy(dtype=float)
    figure, axes = plt.subplots(figsize=(5.0, max(3.2, 0.32 * len(psi_frame))))
    image = axes.imshow(values, cmap="OrRd", aspect="auto", vmin=0.0, vmax=max(0.35, float(np.nanmax(values))))
    axes.set_xticks(range(len(columns)), columns, rotation=0)
    axes.set_yticks(range(len(psi_frame)), psi_frame["feature"], fontsize=8)
    axes.set_title("Population stability index by feature")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axes.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="#ffffff" if value > 0.18 else "#2c2c2a",
            )
    figure.colorbar(image, ax=axes, shrink=0.8, label="PSI")
    axes.grid(False)
    return _save(figure, name)


def plot_vintage(bad_rate_frame: pd.DataFrame, name: str = "vintage_curves.png") -> Path:
    figure, axes = plt.subplots(figsize=(5.6, 4.2))
    axes.bar(
        bad_rate_frame["issue_year"].astype(str),
        bad_rate_frame["observed_default_rate"],
        color=PALETTE["primary"],
        width=0.6,
    )
    for index, row in bad_rate_frame.reset_index(drop=True).iterrows():
        axes.text(
            index,
            float(row["observed_default_rate"]) + 0.004,
            f"{row['observed_default_rate']:.1%}",
            ha="center",
            fontsize=8,
        )
    axes.set_xlabel("Origination vintage (year)")
    axes.set_ylabel("Observed default rate (resolved loans)")
    axes.set_title("Vintage default rates - resolved loans only")
    return _save(figure, name)


def plot_score_distribution(scores: np.ndarray, target: np.ndarray, name: str = "score_distribution.png") -> Path:
    figure, axes = plt.subplots(figsize=(5.6, 4.2))
    axes.hist(scores[target == 0], bins=40, alpha=0.75, color=PALETTE["primary"], label="Non-defaulters")
    axes.hist(scores[target == 1], bins=40, alpha=0.75, color=PALETTE["danger"], label="Defaulters")
    axes.set_xlabel("Credit score (points, higher = lower risk)")
    axes.set_ylabel("Loans")
    axes.set_title("Score separation by outcome - out-of-time sample")
    axes.legend(frameon=False, fontsize=8)
    return _save(figure, name)


def plot_ecl_waterfall(scenario_totals: pd.DataFrame, name: str = "ecl_scenarios.png") -> Path:
    figure, axes = plt.subplots(figsize=(5.8, 4.2))
    labels = scenario_totals["scenario"].tolist()
    stage1 = scenario_totals["stage1_ecl"].to_numpy(dtype=float)
    stage2 = scenario_totals["stage2_ecl"].to_numpy(dtype=float)
    stage3 = scenario_totals["stage3_ecl"].to_numpy(dtype=float)
    positions = np.arange(len(labels))
    axes.bar(positions, stage1, width=0.55, color=PALETTE["primary"], label="Stage 1 (12m ECL)")
    axes.bar(positions, stage2, width=0.55, bottom=stage1, color=PALETTE["secondary"], label="Stage 2 (lifetime ECL)")
    axes.bar(
        positions,
        stage3,
        width=0.55,
        bottom=stage1 + stage2,
        color=PALETTE["danger"],
        label="Stage 3 (credit-impaired)",
    )
    for position, total in zip(positions, stage1 + stage2 + stage3, strict=True):
        axes.text(position, total, f"{total / 1e6:,.2f}m", ha="center", va="bottom", fontsize=8)
    axes.set_xticks(positions, labels)
    axes.set_xlabel("Macro scenario")
    axes.set_ylabel("Expected credit loss")
    axes.set_title("IFRS 9 ECL by stage and macro scenario")
    axes.legend(frameon=False, fontsize=8)
    return _save(figure, name)


def plot_grade_profile(by_grade: pd.DataFrame, name: str = "grade_profile.png") -> Path:
    figure, axes = plt.subplots(figsize=(5.6, 4.2))
    positions = np.arange(len(by_grade))
    axes.bar(positions, by_grade["exposure"] / 1e6, width=0.55, color=PALETTE["muted"], label="Exposure (left)")
    axes.set_ylabel("Exposure (millions)")
    axes.set_xlabel("Internal grade")
    axes.set_xticks(positions, by_grade["grade"])
    second = axes.twinx()
    second.plot(
        positions,
        by_grade["coverage_ratio_pct"],
        color=PALETTE["danger"],
        marker="o",
        linewidth=1.6,
        label="Coverage ratio (right)",
    )
    second.set_ylabel("ECL coverage ratio (%)")
    axes.set_title("Open book by grade: exposure and ECL coverage")
    handles_left, labels_left = axes.get_legend_handles_labels()
    handles_right, labels_right = second.get_legend_handles_labels()
    axes.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left", frameon=False, fontsize=8)
    return _save(figure, name)


def plot_feature_importance(importance: pd.DataFrame, name: str = "feature_importance.png", top_n: int = 15) -> Path:
    subset = importance.head(top_n).iloc[::-1]
    figure, axes = plt.subplots(figsize=(5.4, max(3.4, 0.3 * len(subset))))
    axes.barh(subset["feature"], subset["importance"], color=PALETTE["accent"], height=0.62)
    axes.set_xlabel("Relative importance (gain share)")
    axes.set_title("GBM challenger - feature importance")
    return _save(figure, name)


def plot_iv(iv_frame: pd.DataFrame, name: str = "information_value.png", top_n: int = 18) -> Path:
    subset = iv_frame.head(top_n).iloc[::-1]
    colours = [
        PALETTE["danger"] if value >= 0.5 else PALETTE["primary"] if value >= 0.1 else PALETTE["muted"]
        for value in subset["information_value"]
    ]
    figure, axes = plt.subplots(figsize=(5.4, max(3.4, 0.3 * len(subset))))
    axes.barh(subset["feature"], subset["information_value"], color=colours, height=0.62)
    axes.axvline(0.02, color=PALETTE["muted"], linestyle=":", linewidth=1.0)
    axes.axvline(0.5, color=PALETTE["danger"], linestyle=":", linewidth=1.0)
    axes.set_xlabel("Information value")
    axes.set_title("Feature information value (dotted lines: 0.02 weak, 0.50 leakage-suspicious)")
    return _save(figure, name)


def plot_lift(lift_frame: pd.DataFrame, name: str = "lift_chart.png") -> Path:
    figure, axes = plt.subplots(figsize=(5.6, 4.2))
    positions = np.arange(len(lift_frame))
    axes.bar(positions, lift_frame["lift"], width=0.6, color=PALETTE["primary"])
    axes.axhline(1.0, color=PALETTE["muted"], linestyle="--", linewidth=1.0)
    axes.set_xticks(positions, lift_frame["decile"].astype(str))
    axes.set_xlabel("Risk decile (1 = highest predicted risk)")
    axes.set_ylabel("Lift vs portfolio average")
    axes.set_title("Lift by risk decile - out-of-time sample")
    return _save(figure, name)
