-- ===========================================================================
-- 02_data_quality.sql
-- Purpose : profile the staging layer before any modelling happens. Data
--           quality evidence belongs in version control, not in a notebook
--           cell that nobody re-runs.
-- Dialect : SQLite
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- Missingness by column
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS dq_missing;

CREATE TABLE dq_missing AS
SELECT 'emp_length_yrs' AS column_name,
       SUM(CASE WHEN emp_length_yrs IS NULL THEN 1 ELSE 0 END) AS missing_count,
       COUNT(*) AS total_rows,
       ROUND(100.0 * SUM(CASE WHEN emp_length_yrs IS NULL THEN 1 ELSE 0 END) / COUNT(*), 3) AS missing_pct
FROM staging_loans
UNION ALL
SELECT 'revol_util',
       SUM(CASE WHEN revol_util IS NULL THEN 1 ELSE 0 END),
       COUNT(*),
       ROUND(100.0 * SUM(CASE WHEN revol_util IS NULL THEN 1 ELSE 0 END) / COUNT(*), 3)
FROM staging_loans
UNION ALL
SELECT 'dti',
       SUM(CASE WHEN dti IS NULL THEN 1 ELSE 0 END),
       COUNT(*),
       ROUND(100.0 * SUM(CASE WHEN dti IS NULL THEN 1 ELSE 0 END) / COUNT(*), 3)
FROM staging_loans
UNION ALL
SELECT 'annual_inc',
       SUM(CASE WHEN annual_inc IS NULL THEN 1 ELSE 0 END),
       COUNT(*),
       ROUND(100.0 * SUM(CASE WHEN annual_inc IS NULL THEN 1 ELSE 0 END) / COUNT(*), 3)
FROM staging_loans
UNION ALL
SELECT 'delinq_2yrs',
       SUM(CASE WHEN delinq_2yrs IS NULL THEN 1 ELSE 0 END),
       COUNT(*),
       ROUND(100.0 * SUM(CASE WHEN delinq_2yrs IS NULL THEN 1 ELSE 0 END) / COUNT(*), 3)
FROM staging_loans
UNION ALL
SELECT 'inq_last_6mths',
       SUM(CASE WHEN inq_last_6mths IS NULL THEN 1 ELSE 0 END),
       COUNT(*),
       ROUND(100.0 * SUM(CASE WHEN inq_last_6mths IS NULL THEN 1 ELSE 0 END) / COUNT(*), 3)
FROM staging_loans
UNION ALL
SELECT 'open_acc',
       SUM(CASE WHEN open_acc IS NULL THEN 1 ELSE 0 END),
       COUNT(*),
       ROUND(100.0 * SUM(CASE WHEN open_acc IS NULL THEN 1 ELSE 0 END) / COUNT(*), 3)
FROM staging_loans
UNION ALL
SELECT 'pub_rec',
       SUM(CASE WHEN pub_rec IS NULL THEN 1 ELSE 0 END),
       COUNT(*),
       ROUND(100.0 * SUM(CASE WHEN pub_rec IS NULL THEN 1 ELSE 0 END) / COUNT(*), 3)
FROM staging_loans
UNION ALL
SELECT 'revol_bal',
       SUM(CASE WHEN revol_bal IS NULL THEN 1 ELSE 0 END),
       COUNT(*),
       ROUND(100.0 * SUM(CASE WHEN revol_bal IS NULL THEN 1 ELSE 0 END) / COUNT(*), 3)
FROM staging_loans
UNION ALL
SELECT 'total_acc',
       SUM(CASE WHEN total_acc IS NULL THEN 1 ELSE 0 END),
       COUNT(*),
       ROUND(100.0 * SUM(CASE WHEN total_acc IS NULL THEN 1 ELSE 0 END) / COUNT(*), 3)
FROM staging_loans
ORDER BY missing_pct DESC;

-- ---------------------------------------------------------------------------
-- Duplicate and cardinality checks
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS dq_duplicates;

CREATE TABLE dq_duplicates AS
SELECT
    COUNT(*)                                        AS total_rows,
    COUNT(DISTINCT loan_id)                         AS distinct_loan_ids,
    COUNT(*) - COUNT(DISTINCT loan_id)              AS duplicate_rows,
    COUNT(DISTINCT member_id)                       AS distinct_borrowers,
    ROUND(1.0 * COUNT(*) / COUNT(DISTINCT member_id), 3) AS loans_per_borrower
FROM staging_loans;

-- ---------------------------------------------------------------------------
-- Range / plausibility checks. A non-zero counter is a flag for review.
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS dq_ranges;

CREATE TABLE dq_ranges AS
SELECT 'dti_out_of_range' AS check_name,
       SUM(CASE WHEN dti < 0 OR dti > 100 THEN 1 ELSE 0 END) AS flagged_rows
FROM staging_loans
UNION ALL
SELECT 'fico_out_of_range',
       SUM(CASE WHEN fico_low < 300 OR fico_high > 900 THEN 1 ELSE 0 END)
FROM staging_loans
UNION ALL
SELECT 'int_rate_non_positive',
       SUM(CASE WHEN int_rate <= 0 OR int_rate > 40 THEN 1 ELSE 0 END)
FROM staging_loans
UNION ALL
SELECT 'revol_util_out_of_range',
       SUM(CASE WHEN revol_util < 0 OR revol_util > 150 THEN 1 ELSE 0 END)
FROM staging_loans
UNION ALL
SELECT 'recovery_exceeds_outstanding',
       SUM(CASE WHEN recoveries > out_prncp AND loan_status = 'Charged Off' THEN 1 ELSE 0 END)
FROM staging_loans
UNION ALL
SELECT 'resolved_without_label',
       SUM(CASE WHEN loan_status IN ('Fully Paid', 'Charged Off') AND default_flag IS NULL THEN 1 ELSE 0 END)
FROM staging_loans;

-- ---------------------------------------------------------------------------
-- Label distribution by vintage. This is the table that reveals seasoning:
-- recent vintages carry unresolved ("Current") loans and must not be used to
-- compare vintage quality naively.
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS dq_label_by_vintage;

CREATE TABLE dq_label_by_vintage AS
SELECT
    issue_year,
    COUNT(*)                                                          AS total_loans,
    SUM(CASE WHEN loan_status = 'Fully Paid'  THEN 1 ELSE 0 END)      AS fully_paid,
    SUM(CASE WHEN loan_status = 'Charged Off' THEN 1 ELSE 0 END)      AS charged_off,
    SUM(CASE WHEN loan_status = 'Current'     THEN 1 ELSE 0 END)      AS unresolved,
    ROUND(100.0 * SUM(CASE WHEN loan_status = 'Current' THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                                      AS unresolved_pct,
    ROUND(
        1.0 * SUM(CASE WHEN loan_status = 'Charged Off' THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN loan_status IN ('Fully Paid', 'Charged Off') THEN 1 ELSE 0 END), 0)
    , 4)                                                              AS observed_default_rate
FROM staging_loans
GROUP BY issue_year
ORDER BY issue_year;
