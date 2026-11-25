/* ============================================================================
   01_create_schemas_and_staging.sql
   Project : Customer Churn Intelligence & Retention Action Center
   Engine  : Microsoft SQL Server 2019+ / Express / Developer
   Purpose : Create database schemas, raw staging table, and pipeline QA log.
   ============================================================================ */

IF DB_ID(N'ChurnIntelligenceDB') IS NULL
BEGIN
    CREATE DATABASE ChurnIntelligenceDB;
END;
GO

USE ChurnIntelligenceDB;
GO

IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'stg')  EXEC('CREATE SCHEMA stg');
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'cln')  EXEC('CREATE SCHEMA cln');
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'feat') EXEC('CREATE SCHEMA feat');
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'ml')   EXEC('CREATE SCHEMA ml');
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'bi')   EXEC('CREATE SCHEMA bi');
GO

IF OBJECT_ID(N'stg.telco_customer_raw', N'U') IS NOT NULL
    DROP TABLE stg.telco_customer_raw;
GO

CREATE TABLE stg.telco_customer_raw (
    staging_row_id       INT IDENTITY(1,1) PRIMARY KEY,
    customerID           NVARCHAR(64)  NULL,
    gender               NVARCHAR(32)  NULL,
    SeniorCitizen        NVARCHAR(16)  NULL,
    Partner              NVARCHAR(16)  NULL,
    Dependents           NVARCHAR(16)  NULL,
    tenure               NVARCHAR(32)  NULL,
    PhoneService         NVARCHAR(32)  NULL,
    MultipleLines        NVARCHAR(64)  NULL,
    InternetService      NVARCHAR(64)  NULL,
    OnlineSecurity       NVARCHAR(64)  NULL,
    OnlineBackup         NVARCHAR(64)  NULL,
    DeviceProtection     NVARCHAR(64)  NULL,
    TechSupport          NVARCHAR(64)  NULL,
    StreamingTV          NVARCHAR(64)  NULL,
    StreamingMovies      NVARCHAR(64)  NULL,
    Contract             NVARCHAR(64)  NULL,
    PaperlessBilling     NVARCHAR(16)  NULL,
    PaymentMethod        NVARCHAR(128) NULL,
    MonthlyCharges       NVARCHAR(64)  NULL,
    TotalCharges         NVARCHAR(64)  NULL,
    Churn                NVARCHAR(16)  NULL,
    data_version         NVARCHAR(64)  NOT NULL DEFAULT N'IBM-Telco-v1.0-7043',
    source_file_name     NVARCHAR(255) NOT NULL DEFAULT N'WA_Fn-UseC_-Telco-Customer-Churn.csv',
    ingested_at_utc      DATETIME2(0)  NOT NULL DEFAULT SYSUTCDATETIME()
);
GO

IF OBJECT_ID(N'stg.pipeline_audit_log', N'U') IS NOT NULL
    DROP TABLE stg.pipeline_audit_log;
GO

CREATE TABLE stg.pipeline_audit_log (
    audit_id             INT IDENTITY(1,1) PRIMARY KEY,
    check_stage          NVARCHAR(64)  NOT NULL,
    check_name           NVARCHAR(128) NOT NULL,
    metric_value         DECIMAL(18,4) NOT NULL,
    expected_rule        NVARCHAR(255) NOT NULL,
    check_status         NVARCHAR(16)  NOT NULL CHECK (check_status IN (N'PASS', N'WARN', N'FAIL')),
    notes                NVARCHAR(500) NULL,
    data_version         NVARCHAR(64)  NOT NULL,
    logged_at_utc        DATETIME2(0)  NOT NULL DEFAULT SYSUTCDATETIME()
);
GO

/* Example BULK INSERT command (update file path to local repository folder):
BULK INSERT stg.telco_customer_raw
FROM 'C:\Projects\customer-churn-intelligence-action-center\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv'
WITH (
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '0x0a',
    TABLOCK
);
*/
