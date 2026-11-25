/* ============================================================================
   04_scoring_and_action_tables.sql
   Project : Customer Churn Intelligence & Retention Action Center
   Purpose : Tables written by the Python ML + SHAP scoring pipeline for
             downstream consumption by Power BI and operational care teams.
   ============================================================================ */

USE ChurnIntelligenceDB;
GO

IF OBJECT_ID(N'ml.customer_churn_scores', N'U') IS NOT NULL
    DROP TABLE ml.customer_churn_scores;
GO

CREATE TABLE ml.customer_churn_scores (
    synthetic_customer_id           NVARCHAR(24)  NOT NULL PRIMARY KEY,
    dataset_split                   NVARCHAR(16)  NOT NULL CHECK (dataset_split IN (N'train', N'validation', N'test')),
    churn_probability               DECIMAL(8,6)  NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
    operational_threshold           DECIMAL(8,6)  NOT NULL,
    predicted_high_risk_flag        BIT           NOT NULL,
    risk_band                       NVARCHAR(24)  NOT NULL CHECK (risk_band IN (N'Critical Risk', N'High Risk', N'Moderate Risk', N'Low Risk')),
    monthly_charges_usd             DECIMAL(10,2) NOT NULL,
    monthly_charges_at_risk_proxy   DECIMAL(10,2) NOT NULL,
    expected_monthly_exposure_proxy DECIMAL(10,2) NOT NULL,
    reason_code_1                   NVARCHAR(64)  NOT NULL,
    reason_code_2                   NVARCHAR(64)  NOT NULL,
    reason_code_3                   NVARCHAR(64)  NOT NULL,
    reason_summary_non_causal       NVARCHAR(400) NOT NULL,
    model_name                      NVARCHAR(64)  NOT NULL,
    model_version                   NVARCHAR(32)  NOT NULL,
    data_version                    NVARCHAR(64)  NOT NULL,
    scored_at_utc                   DATETIME2(0)  NOT NULL
);
GO

IF OBJECT_ID(N'ml.retention_action_queue', N'U') IS NOT NULL
    DROP TABLE ml.retention_action_queue;
GO

CREATE TABLE ml.retention_action_queue (
    action_queue_id                 INT IDENTITY(1,1) PRIMARY KEY,
    synthetic_customer_id           NVARCHAR(24)  NOT NULL UNIQUE,
    priority_rank                   INT           NOT NULL,
    priority_tier                   NVARCHAR(24)  NOT NULL,
    risk_band                       NVARCHAR(24)  NOT NULL,
    churn_probability               DECIMAL(8,6)  NOT NULL,
    monthly_charges_usd             DECIMAL(10,2) NOT NULL,
    expected_monthly_exposure_proxy DECIMAL(10,2) NOT NULL,
    priority_score_proxy            DECIMAL(10,2) NOT NULL,
    primary_reason_code             NVARCHAR(64)  NOT NULL,
    reason_summary_non_causal       NVARCHAR(400) NOT NULL,
    recommended_playbook            NVARCHAR(128) NOT NULL,
    playbook_Analytical_Note        NVARCHAR(400) NOT NULL,
    outreach_status                 NVARCHAR(32)  NOT NULL DEFAULT N'Pending Review',
    contact_outcome                 NVARCHAR(64)  NOT NULL DEFAULT N'Not Contacted Yet',
    assigned_team                   NVARCHAR(64)  NOT NULL,
    agent_notes                     NVARCHAR(500) NULL,
    last_updated_utc                DATETIME2(0)  NOT NULL
);
GO

IF OBJECT_ID(N'ml.model_evaluation_registry', N'U') IS NOT NULL
    DROP TABLE ml.model_evaluation_registry;
GO

CREATE TABLE ml.model_evaluation_registry (
    eval_id                         INT IDENTITY(1,1) PRIMARY KEY,
    model_name                      NVARCHAR(80)  NOT NULL,
    imbalance_strategy              NVARCHAR(48)  NOT NULL,
    evaluation_split                NVARCHAR(24)  NOT NULL,
    threshold_used                  DECIMAL(6,4)  NOT NULL,
    roc_auc                         DECIMAL(6,4)  NOT NULL,
    pr_auc                          DECIMAL(6,4)  NOT NULL,
    brier_score                     DECIMAL(6,4)  NOT NULL,
    precision_score                 DECIMAL(6,4)  NOT NULL,
    recall_score                    DECIMAL(6,4)  NOT NULL,
    f1_score                        DECIMAL(6,4)  NOT NULL,
    flagged_rate                    DECIMAL(6,4)  NOT NULL,
    is_selected_champion            BIT           NOT NULL,
    model_version                   NVARCHAR(32)  NOT NULL,
    evaluated_at_utc                DATETIME2(0)  NOT NULL
);
GO
