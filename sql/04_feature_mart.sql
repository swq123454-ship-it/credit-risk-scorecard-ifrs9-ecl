-- ===========================================================================
-- 04_feature_mart.sql
-- Purpose : freeze the modelling mart. This is the single table the Python
--           layer is allowed to read, which keeps the split definition in one
--           auditable place instead of scattered across notebooks.
-- Dialect : SQLite
-- ===========================================================================

DROP TABLE IF EXISTS mart_features;

CREATE TABLE mart_features AS
SELECT
    f.*,
    -- ------------------------------------------------------------------
    -- 12-month default indicator: the target the PD model is actually fitted
    -- on.
    --
    -- This column exists because of a coherence trap that is easy to fall into.
    -- The natural label in a loan-level extract is "did this loan ever charge
    -- off", which is a LIFETIME outcome. Feeding a lifetime label into an ECL
    -- engine that then extrapolates it as a 12-month PD does not merely shift
    -- the answer, it inflates the provision by a large multiple: a 12-month
    -- hazard read as a one-year probability and then compounded over the
    -- remaining term double-counts the horizon.
    --
    -- Defining the target over a fixed 12-month window keeps PD, the term
    -- structure and the ECL horizon on the same clock.
    CASE
        WHEN f.loan_status = 'Charged Off' THEN
            CASE WHEN f.months_to_default IS NOT NULL AND f.months_to_default <= 12 THEN 1.0 ELSE 0.0 END
        WHEN f.loan_status = 'Fully Paid' THEN 0.0
        ELSE NULL
    END AS default_flag_12m,
    CASE
        WHEN f.issue_year BETWEEN 2013 AND 2015 THEN 'train'
        WHEN f.issue_year = 2016               THEN 'test'
        WHEN f.issue_year = 2017               THEN 'oot'
        ELSE 'monitoring'
    END AS split,
    -- Months between origination and the 2018-12-31 reporting date. Used to
    -- season the open book and to build amortisation schedules for ECL.
    ((2018 * 12) + 12) - ((f.issue_year * 12) + f.issue_month) AS months_to_reporting_date
FROM fct_loan AS f;

CREATE INDEX IF NOT EXISTS idx_mart_split ON mart_features (split);

-- ---------------------------------------------------------------------------
-- Vintage summary, restricted to the training vintages.
--
-- Why the restriction matters: a vintage bad-rate table computed over the whole
-- book quietly embeds future information. Restricting it to the training window
-- makes it usable for calibrating scenarios without leaking.
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS vintage_summary;

CREATE TABLE vintage_summary AS
SELECT
    issue_year,
    COUNT(*)                    AS resolved_loans,
    SUM(default_flag)           AS defaults,
    ROUND(AVG(default_flag), 4) AS observed_default_rate,
    ROUND(AVG(out_prncp), 2)    AS avg_exposure_at_default
FROM fct_loan
WHERE loan_status IN ('Fully Paid', 'Charged Off')
  AND issue_year BETWEEN 2013 AND 2015
GROUP BY issue_year
ORDER BY issue_year;

-- ---------------------------------------------------------------------------
-- The open book: loans live at the reporting date. This is the population the
-- IFRS 9 engine actually provisions against - not the resolved population that
-- the PD model was trained on.
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS open_book;

CREATE TABLE open_book AS
SELECT *
FROM mart_features
WHERE loan_status = 'Current'
  AND out_prncp > 0;

CREATE INDEX IF NOT EXISTS idx_open_book_grade ON open_book (grade);

-- ---------------------------------------------------------------------------
-- Portfolio composition by grade, for the concentration discussion.
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS portfolio_by_grade;

CREATE TABLE portfolio_by_grade AS
SELECT
    grade,
    COUNT(*)                    AS loans,
    ROUND(SUM(out_prncp), 2)    AS exposure,
    ROUND(AVG(int_rate), 2)     AS avg_int_rate,
    ROUND(AVG(months_to_reporting_date), 1) AS avg_seasoning_months
FROM open_book
GROUP BY grade
ORDER BY grade;
