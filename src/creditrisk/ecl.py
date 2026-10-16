"""IFRS 9 expected credit loss engine.

This is the part of the project that a credit-risk reviewer will interrogate
hardest, so the design decisions are stated up front.

**What is being provisioned.** Not the training population. The training
population consists of *resolved* loans, whose outcome is known. IFRS 9
provisions against the *open book* at the reporting date, whose outcome is
unknown. Confusing the two is the most common conceptual error in student
projects that claim to "do IFRS 9".

**Three-stage model.**

===========  ==================================  ====================
Stage        Trigger                             ECL horizon
===========  ==================================  ====================
Stage 1      No significant increase in credit   12 months
             risk since origination
Stage 2      SICR: 12-month PD above threshold   Lifetime (remaining
             or 30+ days past due                term)
Stage 3      Credit-impaired: 90+ days past due  Lifetime
===========  ==================================  ====================

**The calculation.**

    ECL = LGD * SUM over t [ marginal_PD(t) * EAD(t) * discount_factor(t) ]

Discounting matters and is frequently skipped. A shortfall expected in month 48
is not worth the same as one expected in month 3.

**What is simplified, and why that is stated rather than hidden.** LGD is
estimated empirically by grade from training vintages rather than modelled as a
workout process; EAD is the contractual amortisation balance without
prepayment modelling; the macro overlay is a documented elasticity rather than a
fitted satellite model. Each of these is flagged in MODEL_DOCUMENTATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg
from .config import EclAssumptions
from .finmath import monthly_hazard_from_annual_pd, outstanding_balance


@dataclass
class EclResult:
    loans: pd.DataFrame
    summary_by_stage: pd.DataFrame
    summary_by_grade: pd.DataFrame
    scenario_totals: pd.DataFrame
    weighted_ecl: float
    total_exposure: float


def scenario_pd_multiplier(
    delta_unemployment_pp: float,
    elasticity: float = cfg.PD_UNEMPLOYMENT_ELASTICITY,
) -> float:
    """PD multiplier from a macro shock: ``exp(elasticity * dU)``.

    The elasticity is illustrative. In a production model it comes from a
    satellite regression of observed default rates on macro factors, estimated
    on a long history. Using a defensible, documented number beats a fabricated
    regression on six annual observations.
    """
    return float(np.exp(elasticity * delta_unemployment_pp))


def assign_stage(
    pd_12m: np.ndarray,
    days_past_due: np.ndarray,
    assumptions: EclAssumptions = cfg.ECL_ASSUMPTIONS,
) -> np.ndarray:
    """Assign IFRS 9 stages: 1 = performing, 2 = SICR, 3 = credit-impaired.

    SICR has two independent triggers, both of which matter:

    * a quantitative trigger (12-month PD above the policy threshold), and
    * a qualitative/backstop trigger (30+ days past due), which is required by
      the standard and cannot be switched off by a low modelled PD.
    """
    pd_12m = np.asarray(pd_12m, dtype=float)
    dpd = np.asarray(days_past_due, dtype=float)
    stage = np.ones_like(pd_12m, dtype=int)
    stage[pd_12m >= assumptions.sicr_pd_threshold] = 2
    stage[dpd >= assumptions.sicr_dpd_threshold] = 2
    stage[dpd >= assumptions.stage3_dpd_threshold] = 3
    return stage


def sicr_sensitivity(
    open_book: pd.DataFrame,
    pd_12m: np.ndarray,
    lgd: np.ndarray,
    thresholds: list[float] | None = None,
    scenario: str = "base",
) -> pd.DataFrame:
    """ECL and Stage 2 population as a function of the SICR PD threshold.

    The answer to "how much of the provision is a policy choice?" should be a
    table, not a paragraph.
    """
    import dataclasses

    thresholds = thresholds or cfg.SICR_SENSITIVITY_THRESHOLDS
    rows: list[dict] = []
    for threshold in thresholds:
        assumptions = dataclasses.replace(cfg.ECL_ASSUMPTIONS, sicr_pd_threshold=float(threshold))
        result = run_ecl_engine(
            open_book,
            pd_12m,
            lgd_values=lgd,
            assumptions=assumptions,
            scenarios={scenario: cfg.MACRO_SCENARIOS[scenario]},
            scenario_weights={scenario: 1.0},
        )
        stage_counts = result.loans["stage"].value_counts()
        rows.append(
            {
                "sicr_pd_threshold": float(threshold),
                "stage1_loans": int(stage_counts.get(1, 0)),
                "stage2_loans": int(stage_counts.get(2, 0)),
                "stage3_loans": int(stage_counts.get(3, 0)),
                "stage2_share_pct": round(100.0 * float(stage_counts.get(2, 0)) / max(len(result.loans), 1), 2),
                "total_ecl": round(float(result.scenario_totals["total_ecl"].iloc[0]), 2),
            }
        )
    table = pd.DataFrame(rows)
    table["ecl_vs_base_pct"] = (100.0 * (table["total_ecl"] / table["total_ecl"].iloc[len(table) // 2] - 1.0)).round(2)
    return table


def horizon_by_stage(
    stage: np.ndarray,
    months_remaining: np.ndarray,
    assumptions: EclAssumptions = cfg.ECL_ASSUMPTIONS,
) -> np.ndarray:
    stage = np.asarray(stage, dtype=int)
    months_remaining = np.asarray(months_remaining, dtype=float)
    horizon = np.where(
        stage == 1,
        np.minimum(assumptions.stage1_horizon_months, months_remaining),
        np.minimum(months_remaining, assumptions.max_lifetime_months),
    )
    return np.maximum(horizon, 0.0)


def loan_level_ecl(
    loan_amnt: np.ndarray,
    int_rate: np.ndarray,
    term_months: np.ndarray,
    months_elapsed: np.ndarray,
    pd_12m: np.ndarray,
    lgd: np.ndarray,
    horizon_months: np.ndarray,
    assumptions: EclAssumptions = cfg.ECL_ASSUMPTIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Discounted ECL per loan, plus undiscounted lifetime PD for reporting.

    Vectorised across loans and looped across months. A direct recursive monthly
    simulation would be clearer per loan and unusable on a real book.
    """
    loan_amnt = np.asarray(loan_amnt, dtype=float)
    int_rate = np.asarray(int_rate, dtype=float)
    term_months = np.asarray(term_months, dtype=float)
    months_elapsed = np.asarray(months_elapsed, dtype=float)
    pd_12m = np.asarray(pd_12m, dtype=float)
    lgd = np.asarray(lgd, dtype=float)
    horizon_months = np.asarray(horizon_months, dtype=float)

    n_loans = loan_amnt.shape[0]
    max_horizon = int(np.max(horizon_months)) if n_loans else 0
    if max_horizon <= 0:
        return np.zeros(n_loans), np.zeros(n_loans)

    hazard = monthly_hazard_from_annual_pd(pd_12m)
    survival = np.ones(n_loans, dtype=float)
    ecl = np.zeros(n_loans, dtype=float)

    discount_rate = np.maximum(int_rate / 100.0, assumptions.discount_rate_floor)

    for month in range(1, max_horizon + 1):
        marginal_pd = survival * hazard
        active = (month <= horizon_months) & ((months_elapsed + month) <= term_months)
        if not np.any(active):
            continue

        elapsed = months_elapsed + month
        exposure = np.zeros(n_loans, dtype=float)
        interesting = np.where(active)[0]
        exposure[interesting] = outstanding_balance(
            loan_amnt[interesting], int_rate[interesting], term_months[interesting], elapsed[interesting]
        )

        df = 1.0 / np.power(1.0 + discount_rate / 12.0, month)
        ecl += lgd * marginal_pd * exposure * df * active
        survival = survival * (1.0 - hazard)

    lifetime_pd = 1.0 - np.power(1.0 - hazard, np.maximum(horizon_months, 0.0))
    return ecl, lifetime_pd


def estimate_lgd_from_warehouse(lgd_table: pd.DataFrame) -> pd.Series:
    """Grade-level empirical LGD, floored and capped.

    Empirical LGD on thin default samples is unstable: a grade with twelve
    defaults can show a recovery rate that is pure noise. The floor/cap limits
    that, and the assumption is recorded in config.
    """
    assumptions = cfg.ECL_ASSUMPTIONS
    table = lgd_table.set_index("grade")["empirical_lgd"].astype(float)
    return table.clip(lower=assumptions.lgd_floor, upper=assumptions.lgd_cap)


def run_ecl_engine(
    open_book: pd.DataFrame,
    pd_12m: np.ndarray,
    lgd_by_grade: pd.Series | None = None,
    lgd_values: np.ndarray | None = None,
    ead_values: np.ndarray | None = None,
    assumptions: EclAssumptions = cfg.ECL_ASSUMPTIONS,
    scenarios: dict[str, float] | None = None,
    scenario_weights: dict[str, float] | None = None,
) -> EclResult:
    """Run the ECL engine across every macro scenario.

    LGD can be supplied either as a modeled per-loan vector (``lgd_values``,
    preferred) or as a grade lookup (``lgd_by_grade``), which is retained as the
    benchmark the model has to beat. ``ead_values`` is reported alongside the
    contractual exposure profile as a cross-check; the ECL projection itself
    always walks the amortisation schedule month by month.
    """
    scenarios = scenarios or cfg.MACRO_SCENARIOS
    scenario_weights = scenario_weights or cfg.SCENARIO_WEIGHTS

    if lgd_values is None and lgd_by_grade is None:
        raise ValueError("Provide either lgd_values (modeled) or lgd_by_grade (benchmark lookup)")

    loans = open_book.copy().reset_index(drop=True)
    pd_12m = np.asarray(pd_12m, dtype=float)

    months_elapsed = loans["months_to_reporting_date"].to_numpy(dtype=float)
    term_months = loans["term_months"].to_numpy(dtype=float)
    months_remaining = np.clip(term_months - months_elapsed, 0.0, None)

    base_stage = assign_stage(pd_12m, loans["days_past_due"].to_numpy(dtype=float), assumptions)
    horizon = horizon_by_stage(base_stage, months_remaining, assumptions)

    if lgd_values is not None:
        lgd = np.asarray(lgd_values, dtype=float)
        lgd_source = "model"
    else:
        mapped = loans["grade"].map(lgd_by_grade)
        lgd = mapped.fillna(float(lgd_by_grade.mean())).to_numpy(dtype=float)
        lgd_source = "grade_lookup"
    lgd = np.clip(lgd, assumptions.lgd_floor, assumptions.lgd_cap)

    loans["pd_12m"] = pd_12m
    loans["pd_12m_base"] = pd_12m
    loans["lgd"] = lgd
    loans["lgd_source"] = lgd_source
    loans["stage"] = base_stage
    loans["ecl_horizon_months"] = horizon
    loans["months_remaining"] = months_remaining
    loans["exposure"] = loans["out_prncp"].to_numpy(dtype=float)
    if ead_values is not None:
        ead_values = np.asarray(ead_values, dtype=float)
        loans["ead_model"] = ead_values
        loans["ead_to_contractual_ratio"] = np.where(
            loans["exposure"] > 0, ead_values / loans["exposure"].replace(0, np.nan), np.nan
        )

    scenario_totals: list[dict] = []
    for scenario, delta_unemployment in scenarios.items():
        multiplier = scenario_pd_multiplier(delta_unemployment)
        scenario_pd = np.clip(pd_12m * multiplier, 1e-6, 0.99)

        scenario_stage = assign_stage(scenario_pd, loans["days_past_due"].to_numpy(dtype=float), assumptions)
        scenario_horizon = horizon_by_stage(scenario_stage, months_remaining, assumptions)

        ecl, lifetime_pd = loan_level_ecl(
            loan_amnt=loans["loan_amnt"].to_numpy(dtype=float),
            int_rate=loans["int_rate"].to_numpy(dtype=float),
            term_months=term_months,
            months_elapsed=months_elapsed,
            pd_12m=scenario_pd,
            lgd=lgd,
            horizon_months=scenario_horizon,
            assumptions=assumptions,
        )

        loans[f"pd_12m_{scenario}"] = scenario_pd
        loans[f"stage_{scenario}"] = scenario_stage
        loans[f"ecl_{scenario}"] = ecl
        loans[f"lifetime_pd_{scenario}"] = lifetime_pd

        scenario_totals.append(
            {
                "scenario": scenario,
                "unemployment_delta_pp": delta_unemployment,
                "pd_multiplier": round(multiplier, 4),
                "weight": scenario_weights.get(scenario, np.nan),
                "total_exposure": round(float(loans["exposure"].sum()), 2),
                "total_ecl": round(float(ecl.sum()), 2),
                "coverage_ratio_pct": round(100.0 * float(ecl.sum()) / max(float(loans["exposure"].sum()), 1e-9), 4),
                "stage1_ecl": round(float(ecl[scenario_stage == 1].sum()), 2),
                "stage2_ecl": round(float(ecl[scenario_stage == 2].sum()), 2),
                "stage3_ecl": round(float(ecl[scenario_stage == 3].sum()), 2),
            }
        )

    scenario_totals_frame = pd.DataFrame(scenario_totals)

    by_stage = (
        loans.groupby("stage")
        .agg(
            loans=("loan_id", "count"),
            exposure=("exposure", "sum"),
            avg_pd_12m=("pd_12m", "mean"),
            avg_lgd=("lgd", "mean"),
            base_ecl=("ecl_base", "sum"),
        )
        .reset_index()
    )
    by_stage["ecl_share_pct"] = 100.0 * by_stage["base_ecl"] / by_stage["base_ecl"].sum()
    by_stage["coverage_ratio_pct"] = 100.0 * by_stage["base_ecl"] / by_stage["exposure"]

    by_grade = (
        loans.groupby("grade")
        .agg(
            loans=("loan_id", "count"),
            exposure=("exposure", "sum"),
            avg_pd_12m=("pd_12m", "mean"),
            avg_lgd=("lgd", "mean"),
            avg_stage=("stage", "mean"),
            base_ecl=("ecl_base", "sum"),
        )
        .reset_index()
    )
    by_grade["coverage_ratio_pct"] = 100.0 * by_grade["base_ecl"] / by_grade["exposure"]

    weighted = 0.0
    weight_sum = 0.0
    for _, row in scenario_totals_frame.iterrows():
        weight = float(row["weight"])
        if np.isnan(weight):
            continue
        weighted += weight * float(row["total_ecl"])
        weight_sum += weight
    weighted_ecl = weighted / weight_sum if weight_sum else float("nan")

    return EclResult(
        loans=loans,
        summary_by_stage=by_stage,
        summary_by_grade=by_grade,
        scenario_totals=scenario_totals_frame,
        weighted_ecl=float(weighted_ecl),
        total_exposure=float(loans["exposure"].sum()),
    )


def staging_transition_matrix(loans: pd.DataFrame) -> pd.DataFrame:
    """How the book migrates from base to severe: the concentration of risk
    migration matters more than the headline ECL increase."""
    if "stage_base" not in loans.columns or "stage_severe" not in loans.columns:
        raise KeyError("Scenario stage columns are missing; run run_ecl_engine first")
    return pd.crosstab(loans["stage_base"], loans["stage_severe"], rownames=["base_stage"], colnames=["severe_stage"])
