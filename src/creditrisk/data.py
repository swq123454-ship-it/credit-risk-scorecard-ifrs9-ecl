"""Data access layer: real LendingClub ingestion plus a synthetic fallback book.

Why a synthetic fallback exists
-------------------------------
The project must be reproducible by a reviewer in one command. The public
LendingClub extract (2007-2018Q4, ~2.9m loans) is distributed through Kaggle,
which requires an authenticated download and cannot be fetched by CI.

Therefore:

* ``--data-mode real`` reads a genuine LendingClub file placed in ``data/raw/``
  and normalises it to the internal schema.
* ``--data-mode synthetic`` (default) generates a *loan-level book with the same
  schema and a known, documented generative process*. Every metric produced in
  synthetic mode validates the pipeline, not the credit market, and the README
  labels it accordingly.

Both modes produce the identical internal schema, so every downstream module
(SQL warehouse, WOE binning, scorecard, GBM, ECL engine) is data-source
agnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .finmath import annuity_payment, outstanding_balance

# ---------------------------------------------------------------------------
# Generative parameters for the synthetic book
# ---------------------------------------------------------------------------
_STATES: list[str] = [
    "CA",
    "TX",
    "NY",
    "FL",
    "IL",
    "PA",
    "OH",
    "GA",
    "NC",
    "MI",
    "NJ",
    "VA",
    "WA",
    "AZ",
    "MA",
    "CO",
    "MD",
    "TN",
    "IN",
    "MO",
]
_STATE_WEIGHTS: np.ndarray = np.array(
    [
        0.13,
        0.09,
        0.08,
        0.08,
        0.05,
        0.05,
        0.04,
        0.04,
        0.04,
        0.04,
        0.03,
        0.03,
        0.03,
        0.03,
        0.03,
        0.03,
        0.02,
        0.02,
        0.02,
        0.02,
    ]
)
_STATE_WEIGHTS = _STATE_WEIGHTS / _STATE_WEIGHTS.sum()

_PURPOSES: list[str] = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "major_purchase",
    "medical",
    "small_business",
    "car",
    "other",
]
_PURPOSE_WEIGHTS: np.ndarray = np.array([0.52, 0.24, 0.07, 0.05, 0.03, 0.03, 0.03, 0.03])

_GRADE_BASE_RATE: dict[str, float] = {
    "A": 6.5,
    "B": 9.0,
    "C": 12.0,
    "D": 15.5,
    "E": 19.5,
    "F": 24.0,
    "G": 27.5,
}
_GRADE_RECOVERY: dict[str, float] = {
    "A": 0.28,
    "B": 0.24,
    "C": 0.20,
    "D": 0.16,
    "E": 0.13,
    "F": 0.10,
    "G": 0.08,
}
_GRADE_LETTERS: str = "ABCDEFG"

#: Vintage effects (log-odds shifts). LendingClub vintages genuinely differed in
#: performance; later 2016 vintages deteriorated before underwriting tightened.
_VINTAGE_SHIFT: dict[int, float] = {
    2013: -0.18,
    2014: -0.05,
    2015: 0.06,
    2016: 0.16,
    2017: 0.04,
    2018: -0.06,
}

#: US unemployment rate (%) used as the macro driver.
_UNEMPLOYMENT: dict[int, float] = {
    2013: 7.4,
    2014: 6.2,
    2015: 5.3,
    2016: 4.9,
    2017: 4.4,
    2018: 3.9,
}

#: Share of recent-vintage loans that have not yet resolved ("Current").
#: These are censored observations: the outcome is unobservable at the
#: reporting date, so they are excluded from PD model training and instead form
#: the open book that the IFRS 9 engine reports on.
_CENSORING_RATE: dict[int, float] = {2013: 0.0, 2014: 0.0, 2015: 0.0, 2016: 0.02, 2017: 0.10, 2018: 0.55}

INTERNAL_SCHEMA: list[str] = [
    "loan_id",
    "member_id",
    "issue_date",
    "term_months",
    "loan_amnt",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length_yrs",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "fico_low",
    "fico_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "loan_status",
    "default_flag",
    "total_pymnt",
    "recoveries",
    "out_prncp",
    "days_past_due",
    "last_pymnt_date",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _solve_intercept(score: np.ndarray, target_mean: float) -> float:
    """Find ``c`` such that ``mean(sigmoid(c + score)) == target_mean``.

    Calibrating the intercept analytically keeps the synthetic book at a
    realistic portfolio default rate regardless of how the risk drivers are
    weighted below.
    """
    low, high = -25.0, 25.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if _sigmoid(mid + score).mean() < target_mean:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def generate_synthetic_loans(
    n_loans: int = 120_000,
    seed: int = cfg.RANDOM_SEED,
    target_default_rate: float = 0.185,
) -> pd.DataFrame:
    """Generate a loan-level book with a documented data-generating process.

    The latent default driver is a linear combination of observable
    underwriting variables plus a vintage effect and idiosyncratic noise, passed
    through a logistic link. Because the driver is known, the pipeline can be
    validated end to end (a model that cannot recover a signal this clean is
    broken), while the noise and vintage effects keep discrimination at a
    realistic level rather than a suspicious 0.99 AUC.
    """
    rng = np.random.default_rng(seed)

    # --- Origination characteristics -------------------------------------
    year_weights = np.array([0.09, 0.11, 0.14, 0.16, 0.20, 0.30])
    issue_year = rng.choice([2013, 2014, 2015, 2016, 2017, 2018], size=n_loans, p=year_weights)
    issue_month = rng.integers(1, 13, size=n_loans)
    issue_day = rng.integers(1, 29, size=n_loans)
    issue_date = pd.to_datetime({"year": issue_year, "month": issue_month, "day": issue_day})

    loan_amnt = np.round(np.exp(rng.normal(np.log(14_000), 0.60, n_loans)).clip(1_000, 40_000) / 100.0) * 100.0
    term_months = np.where(rng.random(n_loans) < 0.72, 36, 60)
    fico_low = np.round(rng.normal(700, 45, n_loans).clip(620, 800) / 5.0) * 5.0
    fico_high = fico_low + 4
    annual_inc = np.exp(rng.normal(np.log(62_000), 0.55, n_loans)).clip(12_000, 500_000)
    dti = rng.normal(18, 8, n_loans).clip(0, 40)
    revol_util = np.where(rng.random(n_loans) < 0.06, np.nan, rng.normal(52, 25, n_loans).clip(0, 120))
    revol_bal = np.exp(rng.normal(np.log(12_000), 0.90, n_loans)).clip(0, 300_000)
    emp_length = np.where(rng.random(n_loans) < 0.08, np.nan, rng.integers(0, 11, n_loans).astype(float))
    delinq_2yrs = rng.poisson(0.32, n_loans).clip(0, 5).astype(float)
    inq_last_6mths = rng.poisson(0.75, n_loans).clip(0, 8).astype(float)
    open_acc = rng.poisson(11, n_loans).clip(2, 40).astype(float)
    pub_rec = rng.poisson(0.22, n_loans).clip(0, 4).astype(float)
    total_acc = rng.poisson(24, n_loans).clip(5, 60).astype(float)
    home_ownership = rng.choice(["MORTGAGE", "RENT", "OWN", "OTHER"], size=n_loans, p=[0.45, 0.38, 0.15, 0.02])
    verification_status = rng.choice(
        ["Not Verified", "Source Verified", "Verified"], size=n_loans, p=[0.35, 0.28, 0.37]
    )
    purpose = rng.choice(_PURPOSES, size=n_loans, p=_PURPOSE_WEIGHTS)
    addr_state = rng.choice(_STATES, size=n_loans, p=_STATE_WEIGHTS)

    # Repeat borrowers: needed so the SQL layer can demonstrate a genuine
    # "prior applications" window function rather than a fabricated one.
    member_pool = rng.integers(10_000_000, 10_000_000 + int(n_loans * 0.72), size=n_loans)

    # --- Latent risk driver ----------------------------------------------
    z_fico = (720.0 - fico_low) / 40.0
    z_dti = (dti - 18.0) / 8.0
    z_revol_util = (np.nan_to_num(revol_util, nan=52.0) - 52.0) / 25.0
    z_revol_bal = (np.log1p(revol_bal) - np.log1p(12_000)) / 1.0
    z_income = -(np.log(annual_inc) - np.log(62_000)) / 0.55
    z_emp = -np.nan_to_num(emp_length, nan=5.0) / 6.0
    z_open_acc = -(open_acc - 11.0) / 6.0
    z_term = (term_months - 36.0) / 24.0
    z_amnt = (np.log(loan_amnt) - np.log(14_000)) / 0.60
    z_home = np.where(home_ownership == "RENT", 0.45, np.where(home_ownership == "MORTGAGE", -0.20, 0.0))
    z_purpose = np.select(
        [
            purpose == "small_business",
            purpose == "medical",
            purpose == "debt_consolidation",
            purpose == "credit_card",
            purpose == "car",
        ],
        [0.55, 0.20, 0.10, -0.12, -0.18],
        default=0.0,
    )
    z_vintage = np.array([_VINTAGE_SHIFT[int(y)] for y in issue_year])
    z_macro = np.array([(_UNEMPLOYMENT[int(y)] - 5.35) for y in issue_year]) * 0.25

    score = (
        0.90 * z_fico
        + 0.35 * z_dti
        + 0.45 * z_revol_util
        + 0.20 * z_revol_bal
        + 0.30 * z_income
        + 0.25 * z_emp
        + 0.15 * z_open_acc
        + 0.30 * z_term
        + 0.15 * z_amnt
        + 0.60 * z_home
        + 0.50 * z_purpose
        + 0.35 * delinq_2yrs
        + 0.30 * inq_last_6mths
        + 0.35 * pub_rec
        + z_vintage
        + z_macro
        + rng.normal(0.0, 0.65, n_loans)
    )

    intercept = _solve_intercept(score, target_default_rate)
    pd_true = _sigmoid(intercept + score)

    # --- Risk grade assigned from the estimated risk, as a lender would ---
    percentile = pd.Series(pd_true).rank(method="first").to_numpy() / n_loans
    grade_idx = np.minimum((percentile * len(_GRADE_LETTERS)).astype(int), len(_GRADE_LETTERS) - 1)
    grade = np.array([_GRADE_LETTERS[i] for i in grade_idx])

    sub_grade = np.empty(n_loans, dtype=object)
    for letter in _GRADE_LETTERS:
        mask = grade == letter
        count = int(mask.sum())
        if count == 0:
            continue
        ranks = pd.Series(pd_true[mask]).rank(method="first").to_numpy()
        band = np.minimum((ranks / count * 5).astype(int), 4) + 1
        sub_grade[mask] = [f"{letter}{b}" for b in band]

    # --- Pricing ----------------------------------------------------------
    base_rate = np.array([_GRADE_BASE_RATE[g] for g in grade])
    int_rate = np.clip(base_rate + (700.0 - fico_low) * 0.022 + rng.normal(0.0, 1.1, n_loans), 5.0, 30.0).round(2)
    installment = np.round(annuity_payment(loan_amnt, int_rate, term_months), 2)

    # --- Observed outcomes ------------------------------------------------
    # Censoring is applied independently of the outcome, which mirrors an
    # immature vintage: the loan may or may not default later, we simply cannot
    # observe it yet. This keeps the resolved sample unbiased.
    censor_rate = np.array([_CENSORING_RATE[int(y)] for y in issue_year])
    censored = rng.random(n_loans) < censor_rate
    defaulted = rng.random(n_loans) < pd_true

    loan_status = np.where(
        censored, cfg.UNSEASONED_LABEL, np.where(defaulted, cfg.DEFAULT_LABEL, cfg.NON_DEFAULT_LABEL)
    )
    default_flag = np.where(censored, np.nan, defaulted.astype(float))

    # --- Exposure and recoveries -----------------------------------------
    reporting_date = pd.Timestamp("2018-12-31")
    months_to_report = (
        ((reporting_date.year - issue_date.dt.year) * 12 + (reporting_date.month - issue_date.dt.month))
        .to_numpy()
        .astype(float)
    )

    # Month in which a defaulted loan charged off.
    default_month = rng.integers(6, 61, n_loans).astype(float)
    default_month = np.minimum(default_month, term_months)

    months_elapsed = np.select(
        [
            loan_status == cfg.UNSEASONED_LABEL,
            loan_status == cfg.DEFAULT_LABEL,
            loan_status == cfg.NON_DEFAULT_LABEL,
        ],
        [
            np.clip(months_to_report, 0, term_months),
            default_month,
            term_months,
        ],
        default=0.0,
    )

    out_prncp = np.round(outstanding_balance(loan_amnt, int_rate, term_months, months_elapsed), 2)
    out_prncp = np.where(loan_status == cfg.NON_DEFAULT_LABEL, 0.0, out_prncp)

    # Recovery depends on grade, on loan size (larger balances recover worse),
    # on term and on borrower income. A worked-out loss that only varied by grade
    # would make a grade-average lookup unbeatable and the LGD model pointless.
    recovery_rate = (
        0.30
        - 0.030 * grade_idx
        - 0.0000045 * loan_amnt
        - 0.0018 * (term_months - 36)
        + 0.0000006 * annual_inc
        + rng.normal(0.0, 0.030, n_loans)
    )
    recovery_rate = np.clip(recovery_rate, 0.02, 0.45)
    recoveries = np.where(loan_status == cfg.DEFAULT_LABEL, np.round(out_prncp * recovery_rate, 2), 0.0)

    total_pymnt = (
        np.where(
            loan_status == cfg.NON_DEFAULT_LABEL,
            installment * term_months,
            installment * months_elapsed,
        )
        + recoveries
    )
    total_pymnt = np.round(total_pymnt, 2)

    # --- Delinquency on the open book (drives IFRS 9 staging) -------------
    dpd_draw = rng.random(n_loans)
    dpd_open = np.select(
        [dpd_draw < 0.918, dpd_draw < 0.978, dpd_draw < 0.993],
        [0, rng.integers(1, 31, n_loans), rng.integers(31, 90, n_loans)],
        default=rng.integers(90, 181, n_loans),
    )
    days_past_due = np.where(loan_status == cfg.UNSEASONED_LABEL, dpd_open, 0).astype(int)

    last_pymnt_date = pd.to_datetime(
        {"year": issue_date.dt.year, "month": issue_date.dt.month, "day": 1}
    ) + pd.to_timedelta((months_elapsed.astype(int) * 30), unit="D")

    frame = pd.DataFrame(
        {
            "loan_id": [f"L{10_000_000 + i}" for i in range(n_loans)],
            "member_id": member_pool,
            "issue_date": issue_date.dt.strftime("%Y-%m-%d"),
            "term_months": term_months.astype(int),
            "loan_amnt": loan_amnt,
            "int_rate": int_rate,
            "installment": installment,
            "grade": grade,
            "sub_grade": sub_grade,
            "emp_length_yrs": emp_length,
            "home_ownership": home_ownership,
            "annual_inc": np.round(annual_inc, 0),
            "verification_status": verification_status,
            "purpose": purpose,
            "addr_state": addr_state,
            "dti": np.round(dti, 2),
            "delinq_2yrs": delinq_2yrs,
            "fico_low": fico_low,
            "fico_high": fico_high,
            "inq_last_6mths": inq_last_6mths,
            "open_acc": open_acc,
            "pub_rec": pub_rec,
            "revol_bal": np.round(revol_bal, 0),
            "revol_util": np.round(revol_util, 2),
            "total_acc": total_acc,
            "loan_status": loan_status,
            "default_flag": default_flag,
            "total_pymnt": total_pymnt,
            "recoveries": recoveries,
            "out_prncp": out_prncp,
            "days_past_due": days_past_due,
            "last_pymnt_date": last_pymnt_date.dt.strftime("%Y-%m-%d"),
        }
    )
    return frame[INTERNAL_SCHEMA]


def generate_synthetic_macro(seed: int = cfg.RANDOM_SEED) -> pd.DataFrame:
    """Macro series used to calibrate the IFRS 9 scenarios."""
    rng = np.random.default_rng(seed + 1)
    years = sorted(_UNEMPLOYMENT)
    unemployment = np.array([_UNEMPLOYMENT[y] for y in years], dtype=float)
    frame = pd.DataFrame(
        {
            "year": years,
            "unemployment_rate": unemployment,
            "unemployment_yoy_change": np.round(np.diff(unemployment, prepend=unemployment[0]), 2),
            "hpi_yoy_growth": np.round(np.array([7.0, 6.0, 5.5, 5.0, 6.2, 5.5]) + rng.normal(0, 0.2, len(years)), 2),
            "gdp_growth": np.round(np.array([2.1, 2.5, 2.9, 1.6, 2.4, 2.9]) + rng.normal(0, 0.15, len(years)), 2),
        }
    )
    return frame


# ---------------------------------------------------------------------------
# Real LendingClub ingestion
# ---------------------------------------------------------------------------
_REAL_COLUMN_MAP: dict[str, str] = {
    "id": "loan_id",
    "member_id": "member_id",
    "loan_amnt": "loan_amnt",
    "term": "term_months",
    "int_rate": "int_rate",
    "installment": "installment",
    "grade": "grade",
    "sub_grade": "sub_grade",
    "emp_length": "emp_length_yrs",
    "home_ownership": "home_ownership",
    "annual_inc": "annual_inc",
    "verification_status": "verification_status",
    "issue_d": "issue_date",
    "purpose": "purpose",
    "addr_state": "addr_state",
    "dti": "dti",
    "delinq_2yrs": "delinq_2yrs",
    "fico_range_low": "fico_low",
    "fico_range_high": "fico_high",
    "inq_last_6mths": "inq_last_6mths",
    "open_acc": "open_acc",
    "pub_rec": "pub_rec",
    "revol_bal": "revol_bal",
    "revol_util": "revol_util",
    "total_acc": "total_acc",
    "loan_status": "loan_status",
    "total_pymnt": "total_pymnt",
    "recoveries": "recoveries",
    "out_prncp": "out_prncp",
    "last_pymnt_d": "last_pymnt_date",
}


def load_real_lendingclub(path) -> pd.DataFrame:
    """Normalise a genuine LendingClub extract to the internal schema.

    Note that LendingClub's own ``loan_status`` vocabulary is richer than the
    binary target (e.g. "Late (31-120 days)", "In Grace Period"). Only
    "Charged Off" and "Fully Paid" are treated as resolved observations, which
    is the conservative choice for PD model development.
    """
    path = path if hasattr(path, "read_csv") else str(path)
    raw = pd.read_csv(path, low_memory=False)
    columns = {k: v for k, v in _REAL_COLUMN_MAP.items() if k in raw.columns}
    frame = raw[list(columns)].rename(columns=columns)

    frame["term_months"] = frame["term_months"].astype(str).str.extract(r"(\d+)").astype(float)
    frame["int_rate"] = frame["int_rate"].astype(str).str.replace("%", "", regex=False).astype(float)
    frame["emp_length_yrs"] = frame["emp_length_yrs"].astype(str).str.extract(r"(\d+)").astype(float)
    frame["issue_date"] = pd.to_datetime(frame["issue_date"], format="%b-%Y", errors="coerce")
    frame = frame.dropna(subset=["issue_date", "loan_amnt", "int_rate", "loan_status"])
    frame["issue_date"] = frame["issue_date"].dt.strftime("%Y-%m-%d")

    status = frame["loan_status"].astype(str)
    frame["default_flag"] = np.where(
        status == "Charged Off",
        1.0,
        np.where(status == "Fully Paid", 0.0, np.nan),
    )
    frame["loan_status"] = np.where(
        status == "Charged Off",
        cfg.DEFAULT_LABEL,
        np.where(status == "Fully Paid", cfg.NON_DEFAULT_LABEL, cfg.UNSEASONED_LABEL),
    )
    frame["days_past_due"] = 0
    if "last_pymnt_date" in frame.columns:
        # LendingClub stores the last payment date as "MMM-YYYY". The SQL layer
        # derives time-to-default from this column, so it must be a real date.
        parsed = pd.to_datetime(frame["last_pymnt_date"], format="%b-%Y", errors="coerce")
        frame["last_pymnt_date"] = parsed.dt.strftime("%Y-%m-%d")
    else:
        frame["last_pymnt_date"] = pd.NaT

    for column in INTERNAL_SCHEMA:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[INTERNAL_SCHEMA]


def load_interim() -> pd.DataFrame:
    """Read the normalised book produced by the ingestion step."""
    frame = pd.read_csv(cfg.INTERIM_LOANS_PATH)
    frame["issue_date"] = pd.to_datetime(frame["issue_date"])
    return frame
