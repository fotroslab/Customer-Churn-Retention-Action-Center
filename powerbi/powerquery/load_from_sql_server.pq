// ============================================================================
// Power Query (M) — SQL Server Express / Developer Connection Template
// Parameters required in Power BI Manage Parameters:
//   pSqlServer   (Text) e.g. "localhost\SQLEXPRESS"
//   pSqlDatabase (Text) e.g. "ChurnIntelligenceDB"
// ============================================================================

let
    Source = Sql.Database(pSqlServer, pSqlDatabase),
    // Replace view_name below for each table:
    //   bi.dim_customer
    //   bi.dim_contract_billing
    //   bi.dim_service_bundle
    //   bi.dim_risk_band
    //   bi.fact_customer_churn_scoring
    //   ml.retention_action_queue
    //   ml.model_evaluation_registry
    //   stg.pipeline_audit_log
    SelectedView = Source{[Schema="bi", Item="fact_customer_churn_scoring"]}[Data],
    TypedColumns = Table.TransformColumnTypes(
        SelectedView,
        {
            {"customer_sk", Int64.Type},
            {"synthetic_customer_id", type text},
            {"contract_billing_key", Int64.Type},
            {"service_bundle_key", Int64.Type},
            {"risk_band_key", Int64.Type},
            {"observed_churn_flag", Int64.Type},
            {"monthly_charges_usd", Currency.Type},
            {"total_charges_usd", Currency.Type},
            {"churn_probability", type number},
            {"operational_threshold", type number},
            {"predicted_high_risk_flag", Int64.Type},
            {"monthly_charges_at_risk_proxy", Currency.Type},
            {"expected_monthly_exposure_proxy", Currency.Type}
        }
    )
in
    TypedColumns
