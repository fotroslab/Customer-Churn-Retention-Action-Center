/* ============================================================================
   02_clean_and_validate.sql
   Project : Customer Churn Intelligence & Retention Action Center
   Purpose : Deduplicate, enforce strict types & domain constraints, handle the
             11 zero-tenure blank TotalCharges records transparently, and create
             non-sensitive pseudonymized customer identifiers.
   ============================================================================ */

USE ChurnIntelligenceDB;
GO

IF OBJECT_ID(N'cln.customer_subscription_clean', N'U') IS NOT NULL
    DROP TABLE cln.customer_subscription_clean;
GO

CREATE TABLE cln.customer_subscription_clean (
    customer_sk                    INT IDENTITY(1,1) PRIMARY KEY,
    synthetic_customer_id          NVARCHAR(24)  NOT NULL UNIQUE,
    source_customer_hash           CHAR(16)      NOT NULL,
    gender                         NVARCHAR(16)  NOT NULL CHECK (gender IN (N'Female', N'Male')),
    senior_citizen                 BIT           NOT NULL,
    partner                        BIT           NOT NULL,
    dependents                     BIT           NOT NULL,
    tenure_months                  INT           NOT NULL CHECK (tenure_months BETWEEN 0 AND 120),
    phone_service                  BIT           NOT NULL,
    multiple_lines                 NVARCHAR(32)  NOT NULL CHECK (multiple_lines IN (N'Yes', N'No', N'No phone service')),
    internet_service               NVARCHAR(32)  NOT NULL CHECK (internet_service IN (N'DSL', N'Fiber optic', N'No')),
    online_security                NVARCHAR(32)  NOT NULL,
    online_backup                  NVARCHAR(32)  NOT NULL,
    device_protection              NVARCHAR(32)  NOT NULL,
    tech_support                   NVARCHAR(32)  NOT NULL,
    streaming_tv                   NVARCHAR(32)  NOT NULL,
    streaming_movies               NVARCHAR(32)  NOT NULL,
    contract_type                  NVARCHAR(32)  NOT NULL CHECK (contract_type IN (N'Month-to-month', N'One year', N'Two year')),
    paperless_billing              BIT           NOT NULL,
    payment_method                 NVARCHAR(64)  NOT NULL CHECK (payment_method IN (
                                       N'Electronic check',
                                       N'Mailed check',
                                       N'Bank transfer (automatic)',
                                       N'Credit card (automatic)'
                                   )),
    monthly_charges                DECIMAL(10,2) NOT NULL CHECK (monthly_charges >= 0),
    total_charges                  DECIMAL(12,2) NOT NULL CHECK (total_charges >= 0),
    was_total_charges_zero_imputed BIT           NOT NULL,
    churn_label                    BIT           NOT NULL,
    churn_status_text              NVARCHAR(8)   NOT NULL CHECK (churn_status_text IN (N'Yes', N'No')),
    data_version                   NVARCHAR(64)  NOT NULL,
    cleaned_at_utc                 DATETIME2(0)  NOT NULL DEFAULT SYSUTCDATETIME()
);
GO

;WITH deduped AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY LTRIM(RTRIM(customerID)) ORDER BY staging_row_id ASC) AS rn,
        ROW_NUMBER() OVER (ORDER BY LTRIM(RTRIM(customerID)) ASC) AS deterministic_seq
    FROM stg.telco_customer_raw
    WHERE NULLIF(LTRIM(RTRIM(customerID)), N'') IS NOT NULL
)
INSERT INTO cln.customer_subscription_clean (
    synthetic_customer_id,
    source_customer_hash,
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure_months,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract_type,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges,
    was_total_charges_zero_imputed,
    churn_label,
    churn_status_text,
    data_version
)
SELECT
    CONCAT(N'SYN-CUST-', RIGHT(CONCAT(N'00000', CAST(deterministic_seq AS NVARCHAR(10))), 5)) AS synthetic_customer_id,
    UPPER(CONVERT(CHAR(16), HASHBYTES('SHA2_256', LTRIM(RTRIM(customerID))), 2)) AS source_customer_hash,
    LTRIM(RTRIM(gender)) AS gender,
    CASE WHEN TRY_CAST(LTRIM(RTRIM(SeniorCitizen)) AS INT) = 1 THEN 1 ELSE 0 END AS senior_citizen,
    CASE WHEN LTRIM(RTRIM(Partner)) = N'Yes' THEN 1 ELSE 0 END AS partner,
    CASE WHEN LTRIM(RTRIM(Dependents)) = N'Yes' THEN 1 ELSE 0 END AS dependents,
    TRY_CAST(LTRIM(RTRIM(tenure)) AS INT) AS tenure_months,
    CASE WHEN LTRIM(RTRIM(PhoneService)) = N'Yes' THEN 1 ELSE 0 END AS phone_service,
    LTRIM(RTRIM(MultipleLines)) AS multiple_lines,
    LTRIM(RTRIM(InternetService)) AS internet_service,
    LTRIM(RTRIM(OnlineSecurity)) AS online_security,
    LTRIM(RTRIM(OnlineBackup)) AS online_backup,
    LTRIM(RTRIM(DeviceProtection)) AS device_protection,
    LTRIM(RTRIM(TechSupport)) AS tech_support,
    LTRIM(RTRIM(StreamingTV)) AS streaming_tv,
    LTRIM(RTRIM(StreamingMovies)) AS streaming_movies,
    LTRIM(RTRIM(Contract)) AS contract_type,
    CASE WHEN LTRIM(RTRIM(PaperlessBilling)) = N'Yes' THEN 1 ELSE 0 END AS paperless_billing,
    LTRIM(RTRIM(PaymentMethod)) AS payment_method,
    TRY_CAST(LTRIM(RTRIM(MonthlyCharges)) AS DECIMAL(10,2)) AS monthly_charges,
    CASE
        WHEN NULLIF(LTRIM(RTRIM(TotalCharges)), N'') IS NULL AND TRY_CAST(LTRIM(RTRIM(tenure)) AS INT) = 0 THEN CAST(0.00 AS DECIMAL(12,2))
        ELSE TRY_CAST(LTRIM(RTRIM(TotalCharges)) AS DECIMAL(12,2))
    END AS total_charges,
    CASE
        WHEN NULLIF(LTRIM(RTRIM(TotalCharges)), N'') IS NULL AND TRY_CAST(LTRIM(RTRIM(tenure)) AS INT) = 0 THEN 1
        ELSE 0
    END AS was_total_charges_zero_imputed,
    CASE WHEN LTRIM(RTRIM(Churn)) = N'Yes' THEN 1 ELSE 0 END AS churn_label,
    LTRIM(RTRIM(Churn)) AS churn_status_text,
    data_version
FROM deduped
WHERE rn = 1;
GO

-- Record data quality checks in stg.pipeline_audit_log
INSERT INTO stg.pipeline_audit_log (check_stage, check_name, metric_value, expected_rule, check_status, notes, data_version)
SELECT N'STAGING', N'raw_row_count', COUNT(*), N'Expected 7043 rows in IBM Telco benchmark',
       CASE WHEN COUNT(*) = 7043 THEN N'PASS' ELSE N'WARN' END,
       N'Total rows ingested into stg.telco_customer_raw', N'IBM-Telco-v1.0-7043'
FROM stg.telco_customer_raw;

INSERT INTO stg.pipeline_audit_log (check_stage, check_name, metric_value, expected_rule, check_status, notes, data_version)
SELECT N'CLEAN', N'zero_tenure_blank_total_charges', SUM(CAST(was_total_charges_zero_imputed AS INT)),
       N'tenure = 0 blank strings mapped deterministically to 0.00',
       N'PASS', N'First-cycle subscribers prior to initial statement generation', N'IBM-Telco-v1.0-7043'
FROM cln.customer_subscription_clean;
GO
