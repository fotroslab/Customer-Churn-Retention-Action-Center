/* ============================================================================
   03_feature_engineering_views.sql
   Project : Customer Churn Intelligence & Retention Action Center
   Purpose : Build strictly row-local, point-in-time observable features.
             NO post-churn signals and NO cross-row population statistics
             (mean/std/quantiles are fit inside Python train folds only)
             to guarantee zero target or train/test data leakage.
   ============================================================================ */

USE ChurnIntelligenceDB;
GO

CREATE OR ALTER VIEW feat.vw_customer_churn_features AS
SELECT
    c.customer_sk,
    c.synthetic_customer_id,
    c.gender,
    CAST(c.senior_citizen AS INT)    AS senior_citizen,
    CAST(c.partner AS INT)           AS partner,
    CAST(c.dependents AS INT)        AS dependents,
    c.tenure_months,
    CASE
        WHEN c.tenure_months <= 6  THEN N'0-6m Onboarding'
        WHEN c.tenure_months <= 12 THEN N'7-12m Early'
        WHEN c.tenure_months <= 24 THEN N'13-24m Developing'
        WHEN c.tenure_months <= 48 THEN N'25-48m Established'
        ELSE N'49m+ Loyal'
    END AS tenure_cohort,
    CAST(c.phone_service AS INT)     AS phone_service,
    c.multiple_lines,
    c.internet_service,
    c.online_security,
    c.online_backup,
    c.device_protection,
    c.tech_support,
    c.streaming_tv,
    c.streaming_movies,
    c.contract_type,
    CAST(c.paperless_billing AS INT) AS paperless_billing,
    c.payment_method,
    c.monthly_charges,
    c.total_charges,
    -- Row-local observable service bundle features (zero leakage across rows)
    (CASE WHEN c.online_security   = N'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.online_backup     = N'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.device_protection = N'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.tech_support      = N'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.streaming_tv      = N'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.streaming_movies  = N'Yes' THEN 1 ELSE 0 END) AS active_addon_count,
    (CASE WHEN c.online_security   = N'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.online_backup     = N'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.device_protection = N'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.tech_support      = N'Yes' THEN 1 ELSE 0 END) AS protection_support_count,
    (CASE WHEN c.streaming_tv      = N'Yes' THEN 1 ELSE 0 END +
     CASE WHEN c.streaming_movies  = N'Yes' THEN 1 ELSE 0 END) AS streaming_addon_count,
    CASE WHEN c.contract_type = N'Month-to-month' THEN 1 ELSE 0 END AS is_month_to_month,
    CASE WHEN c.internet_service = N'Fiber optic' THEN 1 ELSE 0 END AS is_fiber_optic,
    CASE WHEN c.payment_method = N'Electronic check' THEN 1 ELSE 0 END AS is_electronic_check,
    CASE WHEN c.payment_method IN (N'Bank transfer (automatic)', N'Credit card (automatic)')
         THEN 1 ELSE 0 END AS has_auto_pay,
    CAST(
        CASE WHEN c.tenure_months > 0
             THEN c.total_charges / NULLIF(CAST(c.tenure_months AS DECIMAL(10,2)), 0)
             ELSE c.monthly_charges
        END AS DECIMAL(10,2)
    ) AS avg_historical_monthly_spend,
    CAST(
        c.monthly_charges - (
            CASE WHEN c.tenure_months > 0
                 THEN c.total_charges / NULLIF(CAST(c.tenure_months AS DECIMAL(10,2)), 0)
                 ELSE c.monthly_charges
            END
        ) AS DECIMAL(10,2)
    ) AS charge_velocity_delta,
    -- Supervision target ONLY (strictly excluded from model feature matrix X)
    CAST(c.churn_label AS INT)       AS churn_label,
    c.churn_status_text,
    c.data_version,
    c.cleaned_at_utc                 AS feature_snapshot_utc
FROM cln.customer_subscription_clean AS c;
GO
