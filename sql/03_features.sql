-- ===========================================================================
-- 03_features.sql
-- Purpose : build the loan-level feature table using analytic SQL. These
--           features are deliberately engineered in the warehouse rather than
--           in pandas: on a real book the loan table is billions of rows and
--           the aggregation has to happen where the data lives.
--
-- Every window function below is built from origination-time information only.
-- Nothing in this file can see the target, which is what makes it safe.
-- Dialect : SQLite
-- ===========================================================================

DROP TABLE IF EXISTS fct_loan;

CREATE TABLE fct_loan AS
WITH base AS (
    SELECT * FROM staging_loans
),
-- 1. Repeat-borrower ordinal: is this the borrower's 1st, 2nd, 3rd loan with us?
--    Sequential loans are a classic behaviour signal in consumer credit.
borrower_history AS (
    SELECT
        loan_id,
        ROW_NUMBER() OVER (
            PARTITION BY member_id
            ORDER BY issue_date, loan_id
        ) AS borrower_prior_loans
    FROM base
),
-- 2. Geographic concentration: share of the book originated in the borrower's
--    state. Used as an exposure-concentration feature and, in production, as an
--    input to concentration risk limits.
state_volume AS (
    SELECT
        loan_id,
        COUNT(*) OVER (PARTITION BY addr_state) * 1.0
            / (SELECT COUNT(*) FROM base) AS state_volume_share
    FROM base
),
-- 3. Risk rank within grade: lenders grade loans on a coarse scale and then
--    price within grade. A loan priced at the top of its grade is riskier than
--    one priced at the bottom, and the raw grade alone cannot see that.
--    Expressed as a percentile so the scale is stable as the book grows.
grade_rank AS (
    SELECT
        loan_id,
        PERCENT_RANK() OVER (PARTITION BY grade ORDER BY int_rate) AS risk_rank_in_grade
    FROM base
)
SELECT
    b.loan_id,
    b.member_id,
    b.issue_date,
    b.issue_year,
    b.issue_month,
    b.term_months,
    b.loan_amnt,
    b.int_rate,
    b.installment,
    b.grade,
    b.sub_grade,
    b.emp_length_yrs,
    b.home_ownership,
    b.annual_inc,
    b.verification_status,
    b.purpose,
    b.addr_state,
    b.dti,
    b.delinq_2yrs,
    b.fico_low,
    b.fico_high,
    b.inq_last_6mths,
    b.open_acc,
    b.pub_rec,
    b.revol_bal,
    b.revol_util,
    b.total_acc,
    b.loan_status,
    b.default_flag,
    b.total_pymnt,
    b.recoveries,
    b.out_prncp,
    b.days_past_due,
    b.last_pymnt_date,
    -- ----- time to default (workout analytics only) -----------------------
    -- Derived from the last payment date rather than stored, so the same code
    -- path serves the synthetic book and a real LendingClub extract.
    -- NOTE: this column is observable only for a default that has already
    -- happened. It is used to fit the EAD default-timing model and the LGD
    -- workout analysis, and is excluded from every reporting-date model.
    CASE
        WHEN b.loan_status = 'Charged Off' AND b.last_pymnt_date IS NOT NULL
        THEN CAST(ROUND((JULIANDAY(b.last_pymnt_date) - JULIANDAY(b.issue_date)) / 30.44) AS INTEGER)
        ELSE NULL
    END AS months_to_default,
    -- ----- engineered features -------------------------------------------
    h.borrower_prior_loans,
    s.state_volume_share,
    g.risk_rank_in_grade,
    ROUND(b.installment / NULLIF(b.annual_inc / 12.0, 0), 5) AS installment_to_income,
    ROUND(b.loan_amnt   / NULLIF(b.annual_inc, 0), 5)         AS loan_to_income,
    -- ----- derived bands used for reporting and ECL segmentation ---------
    CASE
        WHEN b.annual_inc <  30000 THEN '1_under_30k'
        WHEN b.annual_inc <  60000 THEN '2_30k_60k'
        WHEN b.annual_inc < 100000 THEN '3_60k_100k'
        ELSE '4_over_100k'
    END AS income_band,
    CASE
        WHEN b.dti < 10 THEN '1_under_10'
        WHEN b.dti < 20 THEN '2_10_20'
        WHEN b.dti < 30 THEN '3_20_30'
        ELSE '4_over_30'
    END AS dti_band,
    CASE
        WHEN b.revol_util IS NULL THEN 'unknown'
        WHEN b.revol_util < 0.30 THEN '1_low'
        WHEN b.revol_util < 0.70 THEN '2_medium'
        ELSE '3_high'
    END AS utilisation_band
FROM base AS b
JOIN borrower_history AS h ON h.loan_id = b.loan_id
JOIN state_volume     AS s ON s.loan_id = b.loan_id
JOIN grade_rank       AS g ON g.loan_id = b.loan_id;

CREATE INDEX IF NOT EXISTS idx_fct_loan_member ON fct_loan (member_id);
CREATE INDEX IF NOT EXISTS idx_fct_loan_vintage ON fct_loan (issue_year);

-- ---------------------------------------------------------------------------
-- Empirical recovery / loss-given-default by grade, estimated on the training
-- vintages only. Estimating LGD on the same data used to fit the PD model, and
-- then applying it to that same book, is a common and quiet source of
-- understated provisions.
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS lgd_by_grade;

CREATE TABLE lgd_by_grade AS
SELECT
    grade,
    COUNT(*)                                                  AS defaulted_loans,
    ROUND(AVG(out_prncp), 2)                                  AS avg_exposure_at_default,
    ROUND(AVG(recoveries), 2)                                 AS avg_recoveries,
    ROUND(1.0 - (SUM(recoveries) / NULLIF(SUM(out_prncp), 0)), 4) AS empirical_lgd
FROM fct_loan
WHERE loan_status = 'Charged Off'
  AND issue_year BETWEEN 2013 AND 2015
GROUP BY grade
ORDER BY grade;
