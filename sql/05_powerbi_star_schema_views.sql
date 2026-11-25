/* ============================================================================
   05_powerbi_star_schema_views.sql
   Project : Customer Churn Intelligence & Retention Action Center
   Purpose : Clean Star Schema semantic layer in schema [bi] for Power BI
             Desktop (Import or DirectQuery mode).
   ============================================================================ */

USE ChurnIntelligenceDB;
GO

CREATE OR ALTER VIEW bi.dim_customer AS
SELECT
    f.customer_sk,
    f.synthetic_customer_id,
    f.gender,
    CASE WHEN f.senior_citizen = 1 THEN N'Senior (65+)' ELSE N'Non-Senior' END AS senior_segment,
    CASE WHEN f.partner = 1 THEN N'Yes' ELSE N'No' END AS has_partner,
    CASE WHEN f.dependents = 1 THEN N'Yes' ELSE N'No' END AS has_dependents,
    f.tenure_months,
    f.tenure_cohort
FROM feat.vw_customer_churn_features AS f;
GO

CREATE OR ALTER VIEW bi.dim_contract_billing AS
SELECT
    ROW_NUMBER() OVER (ORDER BY contract_type, payment_method, paperless_billing) AS contract_billing_key,
    contract_type,
    payment_method,
    CASE WHEN paperless_billing = 1 THEN N'Paperless' ELSE N'Paper Statement' END AS billing_delivery,
    CASE WHEN has_auto_pay = 1 THEN N'Automatic Payment' ELSE N'Manual Payment' END AS payment_automation
FROM (
    SELECT DISTINCT contract_type, payment_method, paperless_billing, has_auto_pay
    FROM feat.vw_customer_churn_features
) AS d;
GO

CREATE OR ALTER VIEW bi.dim_service_bundle AS
SELECT
    ROW_NUMBER() OVER (
        ORDER BY internet_service, phone_service, multiple_lines,
                 online_security, online_backup, device_protection,
                 tech_support, streaming_tv, streaming_movies
    ) AS service_bundle_key,
    internet_service,
    CASE WHEN phone_service = 1 THEN N'Yes' ELSE N'No' END AS phone_service,
    multiple_lines,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    active_addon_count,
    protection_support_count,
    streaming_addon_count
FROM (
    SELECT DISTINCT
        internet_service, phone_service, multiple_lines,
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies,
        active_addon_count, protection_support_count, streaming_addon_count
    FROM feat.vw_customer_churn_features
) AS s;
GO

CREATE OR ALTER VIEW bi.dim_risk_band AS
SELECT 1 AS risk_band_key, N'Critical Risk' AS risk_band, 1 AS sort_order, N'>= 0.65'      AS probability_band_desc, N'Within 24h Outreach Queue' AS sla_guidance
UNION ALL
SELECT 2 AS risk_band_key, N'High Risk'     AS risk_band, 2 AS sort_order, N'0.45 - 0.64'  AS probability_band_desc, N'Within 72h Outreach Queue' AS sla_guidance
UNION ALL
SELECT 3 AS risk_band_key, N'Moderate Risk' AS risk_band, 3 AS sort_order, N'Threshold - 0.44' AS probability_band_desc, N'Digital Nurture / Campaign Pool' AS sla_guidance
UNION ALL
SELECT 4 AS risk_band_key, N'Low Risk'      AS risk_band, 4 AS sort_order, N'Below Operational Threshold' AS probability_band_desc, N'No Proactive Intervention' AS sla_guidance;
GO

CREATE OR ALTER VIEW bi.fact_customer_churn_scoring AS
SELECT
    f.customer_sk,
    f.synthetic_customer_id,
    cb.contract_billing_key,
    sb.service_bundle_key,
    rb.risk_band_key,
    f.churn_label                        AS observed_churn_flag,
    f.churn_status_text                  AS observed_churn_text,
    f.monthly_charges                    AS monthly_charges_usd,
    f.total_charges                      AS total_charges_usd,
    f.avg_historical_monthly_spend,
    f.charge_velocity_delta,
    s.dataset_split,
    s.churn_probability,
    s.operational_threshold,
    CAST(s.predicted_high_risk_flag AS INT) AS predicted_high_risk_flag,
    s.risk_band,
    s.monthly_charges_at_risk_proxy,
    s.expected_monthly_exposure_proxy,
    s.reason_code_1,
    s.reason_code_2,
    s.reason_code_3,
    s.reason_summary_non_causal,
    s.model_name,
    s.model_version,
    s.data_version,
    s.scored_at_utc
FROM feat.vw_customer_churn_features AS f
INNER JOIN ml.customer_churn_scores AS s
    ON f.synthetic_customer_id = s.synthetic_customer_id
INNER JOIN bi.dim_contract_billing AS cb
    ON f.contract_type = cb.contract_type
   AND f.payment_method = cb.payment_method
   AND (CASE WHEN f.paperless_billing = 1 THEN N'Paperless' ELSE N'Paper Statement' END) = cb.billing_delivery
INNER JOIN bi.dim_service_bundle AS sb
    ON f.internet_service = sb.internet_service
   AND (CASE WHEN f.phone_service = 1 THEN N'Yes' ELSE N'No' END) = sb.phone_service
   AND f.multiple_lines = sb.multiple_lines
   AND f.online_security = sb.online_security
   AND f.online_backup = sb.online_backup
   AND f.device_protection = sb.device_protection
   AND f.tech_support = sb.tech_support
   AND f.streaming_tv = sb.streaming_tv
   AND f.streaming_movies = sb.streaming_movies
INNER JOIN bi.dim_risk_band AS rb
    ON s.risk_band = rb.risk_band;
GO
