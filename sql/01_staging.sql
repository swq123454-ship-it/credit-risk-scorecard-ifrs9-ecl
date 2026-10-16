-- ===========================================================================
-- 01_staging.sql
-- Purpose : turn the raw ingestion table into a typed, de-duplicated staging
--           layer. No transformations that depend on the target belong here.
-- Dialect : SQLite (also runs on PostgreSQL with a change of the date function)
-- ===========================================================================

DROP TABLE IF EXISTS staging_loans;

CREATE TABLE staging_loans AS
WITH ranked AS (
    SELECT
        raw.*,
        ROW_NUMBER() OVER (
            PARTITION BY raw.loan_id
            ORDER BY raw.issue_date DESC
        ) AS row_rank
    FROM raw_loans AS raw
    WHERE raw.loan_amnt > 0
      AND raw.int_rate IS NOT NULL
      AND raw.fico_low IS NOT NULL
)
SELECT
    loan_id,
    member_id,
    DATE(issue_date)                                       AS issue_date,
    CAST(STRFTIME('%Y', issue_date) AS INTEGER)            AS issue_year,
    CAST(STRFTIME('%m', issue_date) AS INTEGER)            AS issue_month,
    CAST(term_months AS INTEGER)                           AS term_months,
    CAST(loan_amnt AS REAL)                                AS loan_amnt,
    CAST(int_rate AS REAL)                                 AS int_rate,
    CAST(installment AS REAL)                              AS installment,
    grade,
    sub_grade,
    CAST(emp_length_yrs AS REAL)                           AS emp_length_yrs,
    home_ownership,
    CAST(annual_inc AS REAL)                               AS annual_inc,
    verification_status,
    purpose,
    addr_state,
    CAST(dti AS REAL)                                      AS dti,
    CAST(delinq_2yrs AS REAL)                              AS delinq_2yrs,
    CAST(fico_low AS REAL)                                 AS fico_low,
    CAST(fico_high AS REAL)                                AS fico_high,
    CAST(inq_last_6mths AS REAL)                           AS inq_last_6mths,
    CAST(open_acc AS REAL)                                 AS open_acc,
    CAST(pub_rec AS REAL)                                  AS pub_rec,
    CAST(revol_bal AS REAL)                                AS revol_bal,
    CAST(revol_util AS REAL)                               AS revol_util,
    CAST(total_acc AS REAL)                                AS total_acc,
    loan_status,
    CAST(default_flag AS REAL)                             AS default_flag,
    CAST(total_pymnt AS REAL)                              AS total_pymnt,
    CAST(recoveries AS REAL)                               AS recoveries,
    CAST(out_prncp AS REAL)                                AS out_prncp,
    CAST(days_past_due AS INTEGER)                         AS days_past_due,
    last_pymnt_date
FROM ranked
WHERE row_rank = 1;

CREATE INDEX IF NOT EXISTS idx_staging_loans_vintage ON staging_loans (issue_year);
CREATE INDEX IF NOT EXISTS idx_staging_loans_member  ON staging_loans (member_id);

DROP TABLE IF EXISTS staging_macro;

CREATE TABLE staging_macro AS
SELECT
    CAST(year AS INTEGER)                       AS year,
    CAST(unemployment_rate AS REAL)             AS unemployment_rate,
    CAST(unemployment_yoy_change AS REAL)       AS unemployment_yoy_change,
    CAST(hpi_yoy_growth AS REAL)                AS hpi_yoy_growth,
    CAST(gdp_growth AS REAL)                    AS gdp_growth
FROM raw_macro;
