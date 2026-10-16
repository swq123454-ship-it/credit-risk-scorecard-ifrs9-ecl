"""Central configuration for the credit risk / IFRS 9 project.

Every path, modelling window, and business assumption used by the pipeline is
declared here so that a reviewer can audit the model's inputs in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
FIGURE_DIR: Path = OUTPUT_DIR / "figures"
TABLE_DIR: Path = OUTPUT_DIR / "tables"
SQL_DIR: Path = PROJECT_ROOT / "sql"

DB_PATH: Path = PROCESSED_DIR / "credit_risk.db"
INTERIM_LOANS_PATH: Path = INTERIM_DIR / "loans.csv"
INTERIM_MACRO_PATH: Path = INTERIM_DIR / "macro.csv"
METRICS_PATH: Path = OUTPUT_DIR / "metrics.json"

for _directory in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURE_DIR, TABLE_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Out-of-time split design
#
# The book is split chronologically, never randomly. Random splits on credit
# data leak macro conditions across the split and overstate performance.
# ---------------------------------------------------------------------------
TRAIN_YEARS: tuple[int, int] = (2013, 2015)
TEST_YEAR: int = 2016
OOT_YEAR: int = 2017
MONITORING_YEAR: int = 2018  # immaterially seasoned, used for drift monitoring only

# ---------------------------------------------------------------------------
# Target
#
# The PD model is fitted on a fixed 12-month default window
# (``TARGET_COLUMN_12M``), not on the lifetime outcome. The lifetime label is
# retained for vintage analysis and for the LGD/EAD workout models, where the
# full-life outcome is the correct quantity.
# ---------------------------------------------------------------------------
TARGET_COLUMN: str = "default_flag_12m"
TARGET_COLUMN_LIFETIME: str = "default_flag"
DEFAULT_LABEL: str = "Charged Off"
NON_DEFAULT_LABEL: str = "Fully Paid"
UNSEASONED_LABEL: str = "Current"

# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------
ID_COLUMNS: list[str] = ["loan_id", "member_id", "issue_date"]

#: Columns that are only observable *after* origination. Using any of them as a
#: model input produces a look-ahead bias so severe that the model is worthless
#: in production. They are dropped before the feature matrix is assembled.
LEAKAGE_COLUMNS: list[str] = [
    "loan_status",
    "default_flag",
    "total_pymnt",
    "recoveries",
    "out_prncp",
    "days_past_due",
    "last_pymnt_date",
]

NUMERIC_FEATURES: list[str] = [
    "term_months",
    "loan_amnt",
    "int_rate",
    "emp_length_yrs",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_low",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
]

CATEGORICAL_FEATURES: list[str] = [
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
]

#: Features engineered inside the SQL layer (see sql/03_features.sql).
#: They demonstrate relational/analytic-SQL capability and are deliberately
#: built with window functions rather than pandas.
SQL_ENGINEERED_FEATURES: list[str] = [
    "risk_rank_in_grade",
    "state_volume_share",
    "borrower_prior_loans",
    "installment_to_income",
]

MODEL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + SQL_ENGINEERED_FEATURES

# ---------------------------------------------------------------------------
# Scorecard scaling parameters (industry convention)
# ---------------------------------------------------------------------------
SCORECARD_BASE_SCORE: int = 600
SCORECARD_BASE_ODDS: int = 20  # 20:1 good:bad at the base score
SCORECARD_PDO: int = 50  # points to double the odds

# ---------------------------------------------------------------------------
# IFRS 9 assumptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EclAssumptions:
    """Assumptions behind the expected credit loss engine.

    All of these are policy choices, not facts. They are surfaced explicitly so
    that a reviewer can challenge them (this is exactly what a model validation
    function does in a bank).
    """

    #: Lifetime PD horizon is capped at the contractual remaining term.
    max_lifetime_months: int = 60
    #: Empirical LGD is floored/capped to avoid unstable grade-level estimates.
    lgd_floor: float = 0.45
    lgd_cap: float = 0.95
    #: Discount rate applied to projected shortfalls (proxy for the EIR).
    discount_rate_floor: float = 0.05
    #: Stage 1 uses a 12-month horizon; Stage 2/3 use the lifetime horizon.
    stage1_horizon_months: int = 12
    #: Days past due that constitute a significant increase in credit risk.
    sicr_dpd_threshold: int = 30
    #: Loans whose 12-month PD exceeds this level are treated as having had a
    #: significant increase in credit risk (SICR) and move to Stage 2.
    #:
    #: Policy choice, not a fact. In a bank this threshold is calibrated against
    #: the historical Stage 2 population and the ECL sensitivity to it is
    #: disclosed (see ``sicr_sensitivity`` in the report). It is set above the
    #: portfolio's normal PD level so that Stage 2 represents genuine
    #: deterioration rather than most of the book.
    sicr_pd_threshold: float = 0.06
    #: Days past due that trigger Stage 3 (credit-impaired).
    stage3_dpd_threshold: int = 90


ECL_ASSUMPTIONS = EclAssumptions()

#: SICR thresholds tested in the sensitivity analysis. Disclosing how much the
#: provision depends on a policy parameter is standard practice and it is the
#: first question a reviewer asks.
SICR_SENSITIVITY_THRESHOLDS: list[float] = [0.03, 0.045, 0.06, 0.08, 0.12]

#: Macro scenarios expressed as a change in the unemployment rate (percentage
#: points) versus the reporting-date baseline.
#:
#: A downside-only scenario set is not IFRS 9 compliant in spirit: the standard
#: requires a range of outcomes including an upside case, because a portfolio
#: that is only ever provisioned against deterioration systematically overstates
#: the allowance.
MACRO_SCENARIOS: dict[str, float] = {
    "optimistic": -1.5,
    "base": 0.0,
    "adverse": 2.0,
    "severe": 4.5,
}

#: Illustrative elasticity of PD to unemployment: PD_multiplier = exp(beta * du).
#: On real data this coefficient should come from a fitted satellite model
#: (see MODEL_DOCUMENTATION.md, "Known limitations").
PD_UNEMPLOYMENT_ELASTICITY: float = 0.15

#: Scenario probability weights used to derive the probability-weighted ECL.
#: Base case dominates, as in a typical bank's scenario design.
SCENARIO_WEIGHTS: dict[str, float] = {
    "optimistic": 0.10,
    "base": 0.50,
    "adverse": 0.28,
    "severe": 0.12,
}

# ---------------------------------------------------------------------------
# Plot styling (light background: figures are rendered into the README)
# ---------------------------------------------------------------------------
PLOT_STYLE: dict[str, object] = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#5f5e5a",
    "axes.labelcolor": "#2c2c2a",
    "axes.grid": True,
    "grid.color": "#d3d1c7",
    "grid.linewidth": 0.6,
    "text.color": "#2c2c2a",
    "xtick.color": "#5f5e5a",
    "ytick.color": "#5f5e5a",
    "font.size": 10,
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
}

PALETTE: dict[str, str] = {
    "primary": "#185fa5",
    "secondary": "#ba7517",
    "accent": "#0f6e56",
    "danger": "#a32d2d",
    "muted": "#888780",
    "purple": "#534ab7",
}
