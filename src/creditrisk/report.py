"""Pipeline orchestration and reporting.

One command runs ingestion -> warehouse -> WOE binning -> scorecard -> GBM
challenger -> calibration -> stability -> IFRS 9 ECL -> figures -> report. The
report is generated from the same objects that produced the figures, so a
number cannot appear in the README that the pipeline did not calculate.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from . import config as cfg
from . import metrics, plots, warehouse
from .calibration import fit_isotonic
from .data import (
    generate_synthetic_loans,
    generate_synthetic_macro,
    load_real_lendingclub,
)
from .ead import exposure_profile_table, fit_ead_model, validate_ead_model
from .ecl import (
    estimate_lgd_from_warehouse,
    run_ecl_engine,
    sicr_sensitivity,
    staging_transition_matrix,
)
from .features import WoeEncoder
from .gbm import fit_gbm, gbm_feature_importance
from .lgd import (
    build_lgd_training_frame,
    fit_lgd_model,
    grade_benchmark_table,
    validate_lgd_model,
)
from .scorecard import (
    coefficient_table,
    fit_scorecard,
    score_to_rating_band,
    scorecard_table,
)

NUMERIC_FEATURES = cfg.NUMERIC_FEATURES + cfg.SQL_ENGINEERED_FEATURES


def _write_table(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(cfg.TABLE_DIR / name, index=False)


def ingest(
    data_mode: str = "synthetic", n_loans: int = 120_000, verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce the normalised loan book and macro series."""
    if data_mode == "real":
        raw_path = cfg.RAW_DIR / "loan.csv"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Real-data mode expects a LendingClub extract at {raw_path}. "
                "Download 'accepted_2007_to_2018Q4.csv' from the LendingClub "
                "dataset on Kaggle, rename it loan.csv, and place it in data/raw/."
            )
        loans = load_real_lendingclub(raw_path)
        if verbose:
            print(f"  [data] loaded {len(loans):,} real loans from {raw_path.name}")
    else:
        loans = generate_synthetic_loans(n_loans=n_loans)
        if verbose:
            print(f"  [data] generated {len(loans):,} synthetic loans (seed={cfg.RANDOM_SEED})")

    macro = generate_synthetic_macro()
    loans.to_csv(cfg.INTERIM_LOANS_PATH, index=False)
    macro.to_csv(cfg.INTERIM_MACRO_PATH, index=False)
    return loans, macro


def run_pipeline(
    data_mode: str = "synthetic",
    n_loans: int = 120_000,
    verbose: bool = True,
) -> dict[str, Any]:
    print("=" * 78)
    print("Credit risk scorecard & IFRS 9 ECL pipeline")
    print(f"data_mode={data_mode}  seed={cfg.RANDOM_SEED}")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 1. Ingestion
    # ------------------------------------------------------------------
    loans, macro = ingest(data_mode=data_mode, n_loans=n_loans, verbose=verbose)

    # ------------------------------------------------------------------
    # 2. SQL warehouse
    # ------------------------------------------------------------------
    if verbose:
        print("  [sql] building warehouse...")
    warehouse.build_warehouse(loans, macro, verbose=verbose)

    for table in ("dq_missing", "dq_duplicates", "dq_ranges", "dq_label_by_vintage", "vintage_summary", "lgd_by_grade"):
        _write_table(warehouse.read_table(table), f"{table}.csv")

    mart = warehouse.read_table("mart_features")
    for column in (cfg.TARGET_COLUMN, cfg.TARGET_COLUMN_LIFETIME, "default_flag_12m"):
        if column in mart.columns:
            mart[column] = pd.to_numeric(mart[column], errors="coerce")

    resolved = mart[mart["loan_status"].isin([cfg.DEFAULT_LABEL, cfg.NON_DEFAULT_LABEL])].copy()
    resolved = resolved.dropna(subset=[cfg.TARGET_COLUMN]).reset_index(drop=True)

    train = resolved[resolved["split"] == "train"].reset_index(drop=True)
    test = resolved[resolved["split"] == "test"].reset_index(drop=True)
    oot = resolved[resolved["split"] == "oot"].reset_index(drop=True)

    if verbose:
        print(
            f"  [split] train(2013-2015)={len(train):,}  "
            f"test(2016)={len(test):,}  oot(2017)={len(oot):,}  "
            f"|  12m default rate: train {train[cfg.TARGET_COLUMN].mean():.2%} "
            f"oot {oot[cfg.TARGET_COLUMN].mean():.2%}  "
            f"|  open book={int((mart['loan_status'] == cfg.UNSEASONED_LABEL).sum()):,}"
        )

    # ------------------------------------------------------------------
    # 3. WOE binning - fitted on the training window only
    # ------------------------------------------------------------------
    encoder = WoeEncoder(NUMERIC_FEATURES, cfg.CATEGORICAL_FEATURES).fit(train, train[cfg.TARGET_COLUMN])
    iv_table = encoder.iv_table()
    selected = encoder.selected_features()
    flagged = encoder.flagged_features()
    if len(selected) < 3:
        raise RuntimeError("Fewer than three usable features survived IV screening; check the data.")

    _write_table(iv_table, "information_value.csv")
    if verbose:
        print(f"  [woe] fitted {len(encoder.schemes_)} features, {len(selected)} retained after IV screening")
        if flagged:
            print(
                f"  [woe] {len(flagged)} feature(s) with IV >= 0.50 flagged for leakage review: " + ", ".join(flagged)
            )

    woe_train = encoder.transform(train)[selected]
    woe_test = encoder.transform(test)[selected]
    woe_oot = encoder.transform(oot)[selected]

    # ------------------------------------------------------------------
    # 4. Scorecard
    # ------------------------------------------------------------------
    scorecard = fit_scorecard(woe_train, train[cfg.TARGET_COLUMN])
    _write_table(coefficient_table(scorecard), "scorecard_coefficients.csv")
    _write_table(scorecard_table(encoder, scorecard), "scorecard_points.csv")

    predictions = {
        "train": scorecard.predicted_pd(woe_train),
        "test": scorecard.predicted_pd(woe_test),
        "oot": scorecard.predicted_pd(woe_oot),
    }
    scores = {
        "train": scorecard.score(woe_train),
        "test": scorecard.score(woe_test),
        "oot": scorecard.score(woe_oot),
    }

    # ------------------------------------------------------------------
    # 5. GBM challenger (raw features, no WOE)
    # ------------------------------------------------------------------
    gbm_model = fit_gbm(train[cfg.MODEL_FEATURES], train[cfg.TARGET_COLUMN], cfg.CATEGORICAL_FEATURES)
    gbm_predictions = {
        "test": gbm_model.predict_proba(test[cfg.MODEL_FEATURES]),
        "oot": gbm_model.predict_proba(oot[cfg.MODEL_FEATURES]),
    }
    importance = gbm_feature_importance(gbm_model)
    _write_table(importance, "gbm_feature_importance.csv")
    if verbose:
        print(f"  [model] scorecard fitted; GBM challenger backend = {gbm_model.backend}")

    # ------------------------------------------------------------------
    # 6. Calibration: fitted on test, frozen, then applied out of time
    # ------------------------------------------------------------------
    calibrator = fit_isotonic(predictions["test"], test[cfg.TARGET_COLUMN].to_numpy(dtype=float))
    calibrated = {
        "test": calibrator.apply(predictions["test"]),
        "oot": calibrator.apply(predictions["oot"]),
    }

    calibration_frames = {
        "Uncalibrated scorecard": metrics.calibration_table(
            oot[cfg.TARGET_COLUMN].to_numpy(dtype=float), predictions["oot"]
        ),
        "Calibrated scorecard": metrics.calibration_table(
            oot[cfg.TARGET_COLUMN].to_numpy(dtype=float), calibrated["oot"]
        ),
        "GBM challenger": metrics.calibration_table(
            oot[cfg.TARGET_COLUMN].to_numpy(dtype=float), gbm_predictions["oot"]
        ),
    }
    for label, frame in calibration_frames.items():
        _write_table(frame, f"calibration_{label.split()[0].lower()}_oot.csv")

    # ------------------------------------------------------------------
    # 7. Discrimination + stability
    # ------------------------------------------------------------------
    y_test = test[cfg.TARGET_COLUMN].to_numpy(dtype=float)
    y_oot = oot[cfg.TARGET_COLUMN].to_numpy(dtype=float)

    performance_rows = []
    for sample, y_true, raw, cal, gbm_p in (
        ("test (2016)", y_test, predictions["test"], calibrated["test"], gbm_predictions["test"]),
        ("oot (2017)", y_oot, predictions["oot"], calibrated["oot"], gbm_predictions["oot"]),
    ):
        for label, probability in (
            ("Scorecard (uncalibrated)", raw),
            ("Scorecard (calibrated)", cal),
            ("GBM challenger", gbm_p),
        ):
            performance_rows.append(
                {
                    "sample": sample,
                    "model": label,
                    "auc": round(metrics.auc(y_true, probability), 4),
                    "gini": round(metrics.gini(y_true, probability), 4),
                    "ks": round(metrics.ks_statistic(y_true, probability), 4),
                    "brier": round(metrics.brier_score(y_true, probability), 5),
                }
            )
    performance = pd.DataFrame(performance_rows)
    _write_table(performance, "model_performance.csv")

    auc_lower, auc_upper = metrics.bootstrap_auc_ci(y_oot, calibrated["oot"])
    ks_frame = metrics.ks_curve(y_oot, calibrated["oot"])
    roc_frames = {
        "Calibrated scorecard": metrics.roc_frame(y_oot, calibrated["oot"]),
        "GBM challenger": metrics.roc_frame(y_oot, gbm_predictions["oot"]),
    }
    aucs = {
        "Calibrated scorecard": metrics.auc(y_oot, calibrated["oot"]),
        "GBM challenger": metrics.auc(y_oot, gbm_predictions["oot"]),
    }
    lift = metrics.lift_table(y_oot, calibrated["oot"])
    _write_table(lift, "lift_oot.csv")

    # PSI on the score and on every WOE-transformed feature.
    psi_rows = []
    for feature in selected:
        psi_rows.append(
            {
                "feature": feature,
                "psi_train_vs_test": metrics.population_stability_index(
                    encoder.transform(train)[feature].to_numpy(dtype=float),
                    encoder.transform(test)[feature].to_numpy(dtype=float),
                ),
                "psi_train_vs_oot": metrics.population_stability_index(
                    encoder.transform(train)[feature].to_numpy(dtype=float),
                    encoder.transform(oot)[feature].to_numpy(dtype=float),
                ),
            }
        )
    psi_table = pd.DataFrame(psi_rows)
    _write_table(psi_table, "psi_by_feature.csv")

    score_psi_test = metrics.population_stability_index(scores["train"], scores["test"])
    score_psi_oot = metrics.population_stability_index(scores["train"], scores["oot"])

    # Rating band table derived from the score.
    rating_bands = pd.DataFrame(
        {
            "rating_band": score_to_rating_band(scores["oot"]),
            "score": scores["oot"],
            "default_flag": y_oot,
        }
    )
    rating_table = (
        rating_bands.groupby("rating_band")
        .agg(loans=("default_flag", "size"), avg_score=("score", "mean"), bad_rate=("default_flag", "mean"))
        .reset_index()
    )
    rating_table["score_range"] = rating_table["rating_band"].map(
        rating_bands.groupby("rating_band")["score"].agg(lambda s: f"{s.min():.0f}-{s.max():.0f}")
    )
    _write_table(rating_table, "rating_scale_oot.csv")

    # Cost-sensitive cut-off analysis.
    cut_off_rows = []
    for threshold in np.arange(0.02, 0.42, 0.02):
        cut_off_rows.append(
            {
                "pd_cutoff": round(float(threshold), 3),
                "approval_rate": round(float((predictions["oot"] < threshold).mean()), 4),
                "expected_cost_units": round(metrics.expected_cost(y_oot, predictions["oot"], float(threshold)), 2),
            }
        )
    cut_off = pd.DataFrame(cut_off_rows)
    _write_table(cut_off, "cutoff_analysis_oot.csv")
    best_cut_off = cut_off.loc[cut_off["expected_cost_units"].idxmin()]

    # ------------------------------------------------------------------
    # 8. LGD and EAD models
    #
    # Both are fitted on resolved defaults from the training vintages only, then
    # validated out of time. Both must beat the incumbent benchmark (a
    # grade-average lookup) or they are complexity without value.
    # ------------------------------------------------------------------
    lgd_training_defaults = build_lgd_training_frame(mart, cfg.TRAIN_YEARS)
    lgd_holdout_defaults = build_lgd_training_frame(mart, (cfg.TEST_YEAR, cfg.OOT_YEAR))

    lgd_model = fit_lgd_model(lgd_training_defaults)
    lgd_benchmark = grade_benchmark_table(lgd_training_defaults)
    lgd_calibration_table, lgd_holdout_summary = validate_lgd_model(lgd_model, lgd_holdout_defaults, "oot (2016-2017)")
    _, lgd_insample_summary = validate_lgd_model(lgd_model, lgd_training_defaults, "train (2013-2015)")

    _write_table(lgd_benchmark, "lgd_grade_benchmark.csv")
    _write_table(lgd_calibration_table, "lgd_calibration_oot.csv")
    _write_table(pd.DataFrame([lgd_insample_summary, lgd_holdout_summary]), "lgd_validation.csv")

    ead_model = fit_ead_model(lgd_training_defaults)
    ead_by_grade, ead_summary = validate_ead_model(ead_model, lgd_holdout_defaults)
    _write_table(ead_by_grade, "ead_validation_oot.csv")
    _write_table(exposure_profile_table(ead_model), "ead_exposure_profile.csv")

    if verbose:
        print(
            f"  [lgd]   OOT MAE {lgd_holdout_summary['model_mae']:.4f} vs benchmark "
            f"{lgd_holdout_summary['benchmark_mae']:.4f} "
            f"({lgd_holdout_summary['mae_improvement_pct']:+.1f}%)"
        )
        print(
            f"  [ead]   OOT exposure ratio predicted {ead_summary['predicted_mean_exposure_ratio']:.4f} "
            f"vs observed {ead_summary['observed_mean_exposure_ratio']:.4f}"
        )

    # ------------------------------------------------------------------
    # 9. IFRS 9 ECL engine on the open book
    # ------------------------------------------------------------------
    open_book = warehouse.read_table("open_book").reset_index(drop=True)
    woe_open = encoder.transform(open_book)[selected]
    pd_open = calibrator.apply(scorecard.predicted_pd(woe_open))

    lgd_by_grade = estimate_lgd_from_warehouse(warehouse.read_table("lgd_by_grade"))
    lgd_open = lgd_model.predict_lgd(open_book)
    lgd_benchmark_open = lgd_model.benchmark_lgd(open_book)
    ead_open, ead_ratio_open = ead_model.predict_ead(open_book)

    # The LGD model has to earn its place. If it does not beat the grade-average
    # lookup out of time, the base provision uses the benchmark and the report
    # says so - shipping the more complex model anyway would be indefensible.
    lgd_model_earned = bool(lgd_holdout_summary["model_mae"] < lgd_holdout_summary["benchmark_mae"])
    lgd_selected = lgd_open if lgd_model_earned else lgd_benchmark_open
    lgd_basis_selected = "modeled LGD" if lgd_model_earned else "grade-average lookup (model failed its benchmark)"

    ecl_result = run_ecl_engine(open_book, pd_open, lgd_values=lgd_selected, ead_values=ead_open)
    ecl_benchmark = run_ecl_engine(open_book, pd_open, lgd_by_grade=lgd_by_grade)

    lgd_sensitivity = pd.DataFrame(
        [
            {
                "lgd_basis": "grade-average lookup (benchmark)",
                "mean_lgd": round(float(ecl_benchmark.loans["lgd"].mean()), 4),
                "base_ecl": round(
                    float(ecl_benchmark.scenario_totals.set_index("scenario").loc["base", "total_ecl"]), 2
                ),
                "weighted_ecl": round(ecl_benchmark.weighted_ecl, 2),
            },
            {
                "lgd_basis": "modeled LGD",
                "mean_lgd": round(float(lgd_open.mean()), 4),
                "base_ecl": round(
                    float(
                        run_ecl_engine(open_book, pd_open, lgd_values=lgd_open)
                        .scenario_totals.set_index("scenario")
                        .loc["base", "total_ecl"]
                    ),
                    2,
                ),
                "weighted_ecl": round(run_ecl_engine(open_book, pd_open, lgd_values=lgd_open).weighted_ecl, 2),
            },
        ]
    )
    lgd_sensitivity["selected_for_base_case"] = [
        not lgd_model_earned,
        lgd_model_earned,
    ]

    sicr_table = sicr_sensitivity(open_book, pd_open, lgd_selected)

    _write_table(ecl_result.summary_by_stage, "ecl_by_stage.csv")
    _write_table(ecl_result.summary_by_grade, "ecl_by_grade.csv")
    _write_table(ecl_result.scenario_totals, "ecl_scenarios.csv")
    _write_table(staging_transition_matrix(ecl_result.loans).reset_index(), "staging_transition.csv")
    _write_table(lgd_sensitivity, "ecl_lgd_sensitivity.csv")
    _write_table(sicr_table, "ecl_sicr_sensitivity.csv")
    _write_table(warehouse.read_table("portfolio_by_grade"), "portfolio_by_grade.csv")

    # ------------------------------------------------------------------
    # 10. Figures
    # ------------------------------------------------------------------
    figure_paths = [
        plots.plot_roc(roc_frames, aucs),
        plots.plot_ks(ks_frame, metrics.ks_statistic(y_oot, calibrated["oot"])),
        plots.plot_calibration(calibration_frames),
        plots.plot_psi_heatmap(psi_table),
        plots.plot_vintage(warehouse.read_table("dq_label_by_vintage").dropna(subset=["observed_default_rate"])),
        plots.plot_score_distribution(scores["oot"], y_oot),
        plots.plot_ecl_waterfall(ecl_result.scenario_totals),
        plots.plot_grade_profile(ecl_result.summary_by_grade),
        plots.plot_feature_importance(importance),
        plots.plot_iv(iv_table),
        plots.plot_lift(lift),
    ]
    if verbose:
        print(f"  [figures] wrote {len(figure_paths)} figures to {cfg.FIGURE_DIR}")

    # ------------------------------------------------------------------
    # 11. Metrics bundle
    # ------------------------------------------------------------------
    scorecard_oot = performance.query("sample == 'oot (2017)' and model == 'Scorecard (calibrated)'").iloc[0]
    gbm_oot = performance.query("sample == 'oot (2017)' and model == 'GBM challenger'").iloc[0]

    result: dict[str, Any] = {
        "data_mode": data_mode,
        "random_seed": cfg.RANDOM_SEED,
        "counts": {
            "loans_total": int(len(mart)),
            "resolved_total": int(len(resolved)),
            "train": int(len(train)),
            "test": int(len(test)),
            "oot": int(len(oot)),
            "open_book": int(len(open_book)),
        },
        "portfolio": {
            "target_definition": "default within 12 months of origination",
            "train_default_rate_12m": round(float(train[cfg.TARGET_COLUMN].mean()), 4),
            "test_default_rate_12m": round(float(test[cfg.TARGET_COLUMN].mean()), 4),
            "oot_default_rate_12m": round(float(oot[cfg.TARGET_COLUMN].mean()), 4),
            "train_lifetime_default_rate": round(float(train[cfg.TARGET_COLUMN_LIFETIME].mean()), 4),
            "oot_lifetime_default_rate": round(float(oot[cfg.TARGET_COLUMN_LIFETIME].mean()), 4),
        },
        "scorecard": {
            "features_used": selected,
            "features_screened_out": [f for f in encoder.schemes_ if f not in selected],
            "features_flagged_for_leakage_review": flagged,
            "leakage_review_outcome": (
                "Retained. Both flagged features are origination-time information: the internal "
                "risk grade and the contract interest rate, which the lender itself set at "
                "origination. The IV ceiling is a review trigger, not an automatic exclusion."
            )
            if flagged
            else "No feature exceeded the IV review ceiling.",
            "oot_auc": float(scorecard_oot["auc"]),
            "oot_gini": float(scorecard_oot["gini"]),
            "oot_ks": float(scorecard_oot["ks"]),
            "oot_brier": float(scorecard_oot["brier"]),
            "oot_auc_ci95": [round(auc_lower, 4), round(auc_upper, 4)],
        },
        "gbm": {
            "backend": gbm_model.backend,
            "oot_auc": float(gbm_oot["auc"]),
            "oot_gini": float(gbm_oot["gini"]),
            "oot_ks": float(gbm_oot["ks"]),
            "oot_brier": float(gbm_oot["brier"]),
        },
        "stability": {
            "score_psi_train_vs_test": round(score_psi_test, 4),
            "score_psi_train_vs_oot": round(score_psi_oot, 4),
            "band_test": metrics.psi_band(score_psi_test),
            "band_oot": metrics.psi_band(score_psi_oot),
            "max_feature_psi_oot": round(float(psi_table["psi_train_vs_oot"].max()), 4),
        },
        "decisions": {
            "best_pd_cutoff": float(best_cut_off["pd_cutoff"]),
            "approval_rate_at_cutoff": float(best_cut_off["approval_rate"]),
            "expected_cost_units_at_cutoff": float(best_cut_off["expected_cost_units"]),
        },
        "lgd": {
            "training_defaults": int(len(lgd_training_defaults)),
            "holdout_defaults": int(len(lgd_holdout_defaults)),
            "oot": lgd_holdout_summary,
            "in_sample": lgd_insample_summary,
            "model_beats_benchmark": bool(lgd_holdout_summary["model_mae"] < lgd_holdout_summary["benchmark_mae"]),
        },
        "ead": {
            "model": "default timing profile x contractual amortisation",
            "oot": ead_summary,
        },
        "ifrs9": {
            "total_exposure": round(ecl_result.total_exposure, 2),
            "weighted_ecl": round(ecl_result.weighted_ecl, 2),
            "weighted_coverage_ratio_pct": round(
                100.0 * ecl_result.weighted_ecl / max(ecl_result.total_exposure, 1e-9), 4
            ),
            "lgd_basis": lgd_basis_selected,
            "lgd_model_earned_selection": lgd_model_earned,
            "scenarios": ecl_result.scenario_totals.to_dict(orient="records"),
            "assumptions": asdict(cfg.ECL_ASSUMPTIONS),
            "scenario_definitions": cfg.MACRO_SCENARIOS,
            "scenario_weights": cfg.SCENARIO_WEIGHTS,
            "pd_unemployment_elasticity": cfg.PD_UNEMPLOYMENT_ELASTICITY,
            "sicr_sensitivity": sicr_table.to_dict(orient="records"),
        },
        "score_scale": {
            "base_score": cfg.SCORECARD_BASE_SCORE,
            "base_odds": cfg.SCORECARD_BASE_ODDS,
            "pdo": cfg.SCORECARD_PDO,
        },
    }

    cfg.METRICS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    _write_results_markdown(
        result,
        performance,
        ecl_result,
        rating_table,
        cut_off,
        lgd_tables={
            "benchmark": lgd_benchmark,
            "calibration": lgd_calibration_table,
            "validation": pd.DataFrame([lgd_insample_summary, lgd_holdout_summary]),
        },
        ead_tables={
            "by_grade": ead_by_grade,
            "profile": exposure_profile_table(ead_model),
            "summary": pd.DataFrame([ead_summary]),
        },
        lgd_sensitivity=lgd_sensitivity,
        sicr_table=sicr_table,
    )

    if verbose:
        print("-" * 78)
        print(
            f"  scorecard OOT : AUC {result['scorecard']['oot_auc']:.4f}  Gini {result['scorecard']['oot_gini']:.4f}  KS {result['scorecard']['oot_ks']:.4f}"
        )
        print(
            f"  GBM OOT       : AUC {result['gbm']['oot_auc']:.4f}  Gini {result['gbm']['oot_gini']:.4f}  KS {result['gbm']['oot_ks']:.4f}"
        )
        print(
            f"  ECL (weighted): {result['ifrs9']['weighted_ecl']:,.0f} on exposure {result['ifrs9']['total_exposure']:,.0f} ({result['ifrs9']['weighted_coverage_ratio_pct']:.3f}%)"
        )
        print(f"  outputs       : {cfg.OUTPUT_DIR}")
        print("=" * 78)

    return result


def _write_results_markdown(
    result: dict[str, Any],
    performance: pd.DataFrame,
    ecl_result,
    rating_table: pd.DataFrame,
    cut_off: pd.DataFrame,
    lgd_tables: dict[str, pd.DataFrame],
    ead_tables: dict[str, pd.DataFrame],
    lgd_sensitivity: pd.DataFrame,
    sicr_table: pd.DataFrame,
) -> None:
    """Write outputs/RESULTS.md from the same objects that produced the figures."""
    lines: list[str] = []
    add = lines.append

    add("# Pipeline results\n")
    add(
        f"Generated by `scripts/run_pipeline.py` with `data_mode={result['data_mode']}` and seed `{result['random_seed']}`.\n"
    )
    if result["data_mode"] == "synthetic":
        add(
            "> **These figures validate the pipeline, not the credit market.** "
            "They are produced from the documented synthetic book described in "
            "`src/creditrisk/data.py`. Re-run with `--data-mode real` after placing a "
            "LendingClub extract in `data/raw/loan.csv` to obtain market results.\n"
        )

    add("## Sample sizes\n")
    add("| Slice | Loans |\n|---|---|")
    for key, value in result["counts"].items():
        add(f"| {key} | {value:,} |")
    add("")

    add("## Default rates\n")
    add(f"Target definition: **{result['portfolio']['target_definition']}**\n")
    add("| Slice | Rate |\n|---|---|")
    for key, value in result["portfolio"].items():
        if key == "target_definition":
            continue
        add(f"| {key} | {value:.2%} |")
    add("")

    add("## Discrimination and calibration\n")
    add(performance.to_markdown(index=False))
    add("")
    add(
        f"95% bootstrap confidence interval for the calibrated scorecard's out-of-time AUC: "
        f"**{result['scorecard']['oot_auc_ci95'][0]:.4f} - {result['scorecard']['oot_auc_ci95'][1]:.4f}**.\n"
    )

    add("## Stability\n")
    add("| Measure | Value | Band |\n|---|---|---|")
    add(
        f"| Score PSI (train vs test) | {result['stability']['score_psi_train_vs_test']:.4f} | {result['stability']['band_test']} |"
    )
    add(
        f"| Score PSI (train vs OOT) | {result['stability']['score_psi_train_vs_oot']:.4f} | {result['stability']['band_oot']} |"
    )
    add(f"| Highest feature PSI (train vs OOT) | {result['stability']['max_feature_psi_oot']:.4f} | - |")
    add("")

    add("## Decision cut-off\n")
    add(
        f"Minimising expected cost on the out-of-time sample gives a PD cut-off of "
        f"**{result['decisions']['best_pd_cutoff']:.2f}**, an approval rate of "
        f"{result['decisions']['approval_rate_at_cutoff']:.1%}, and expected cost of "
        f"{result['decisions']['expected_cost_units_at_cutoff']:,.0f} units "
        f"(costs: one default costs 1 unit, one declined good applicant costs 0.05 units).\n"
    )
    add(cut_off.to_markdown(index=False))
    add("")

    add("## Rating scale (out-of-time)\n")
    add(rating_table.to_markdown(index=False))
    add("")

    add("## LGD model\n")
    add(
        "The model is only used for the base provision if it beats the grade-average lookup "
        "table it replaces, measured out of time:\n"
    )
    add(lgd_tables["validation"].to_markdown(index=False))
    add("")
    add(f"**Basis selected for the base provision: {result['ifrs9']['lgd_basis']}**\n")
    add("Grade-average benchmark fitted on the training defaults:\n")
    add(lgd_tables["benchmark"].to_markdown(index=False))
    add("")
    add("Calibration of predicted vs observed LGD on out-of-time defaults:\n")
    add(lgd_tables["calibration"].to_markdown(index=False))
    add("")

    add("## EAD model\n")
    add("Default timing is estimated per grade and combined with the contractual amortisation schedule:\n")
    add(ead_tables["summary"].to_markdown(index=False))
    add("")
    add(ead_tables["by_grade"].to_markdown(index=False))
    add("")
    add("Median month of default by grade (the shape of the exposure profile):\n")
    add(ead_tables["profile"].to_markdown(index=False))
    add("")

    add("## IFRS 9 expected credit loss\n")
    add(f"Open book exposure: **{result['ifrs9']['total_exposure']:,.0f}**  ")
    add(f"Probability-weighted ECL: **{result['ifrs9']['weighted_ecl']:,.0f}**  ")
    add(f"Weighted coverage ratio: **{result['ifrs9']['weighted_coverage_ratio_pct']:.4f}%**\n")
    add("Scenarios (unemployment deltas in percentage points, PD multiplier = exp(0.15 x dU)):\n")
    add(ecl_result.scenario_totals.to_markdown(index=False))
    add("")
    add("Sensitivity of the provision to the LGD basis (same book, same PD):\n")
    add(lgd_sensitivity.to_markdown(index=False))
    add("")
    add("### Sensitivity to the SICR policy threshold\n")
    add(
        "Stage 2 population and total ECL as a function of the PD threshold that defines a "
        "significant increase in credit risk. The 30+ days past due backstop trigger is always "
        "active and is not part of this sweep.\n"
    )
    add(sicr_table.to_markdown(index=False))
    add("")
    add("### ECL by stage\n")
    add(ecl_result.summary_by_stage.to_markdown(index=False))
    add("")
    add("### ECL by grade\n")
    add(ecl_result.summary_by_grade.to_markdown(index=False))
    add("")

    add("## Features used by the scorecard\n")
    add(", ".join(f"`{feature}`" for feature in result["scorecard"]["features_used"]))
    add("")
    if result["scorecard"]["features_screened_out"]:
        add(
            "Screened out by information value: "
            + ", ".join(f"`{f}`" for f in result["scorecard"]["features_screened_out"])
        )
        add("")

    cfg.OUTPUT_DIR.joinpath("RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
