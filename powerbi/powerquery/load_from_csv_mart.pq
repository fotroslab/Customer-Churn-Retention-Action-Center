// ============================================================================
// Power Query (M) — Zero-Database Quick Load from Star Schema CSV Folder
// Parameter required in Power BI Manage Parameters:
//   pMartFolderPath (Text) e.g. "C:\Projects\customer-churn-intelligence-action-center\data\powerbi_mart"
// ============================================================================

let
    LoadMartCsv = (fileName as text) as table =>
        let
            RawCsv = Csv.Document(
                File.Contents(pMartFolderPath & "\" & fileName),
                [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
            ),
            Promoted = Table.PromoteHeaders(RawCsv, [PromoteAllScalars=true])
        in
            Promoted,
    // Example invocation for fact_customer_churn_scoring.csv:
    FactScoring = Table.TransformColumnTypes(
        LoadMartCsv("fact_customer_churn_scoring.csv"),
        {
            {"customer_sk", Int64.Type},
            {"contract_billing_key", Int64.Type},
            {"service_bundle_key", Int64.Type},
            {"risk_band_key", Int64.Type},
            {"observed_churn_flag", Int64.Type},
            {"monthly_charges_usd", type number},
            {"total_charges_usd", type number},
            {"churn_probability", type number},
            {"operational_threshold", type number},
            {"predicted_high_risk_flag", Int64.Type},
            {"monthly_charges_at_risk_proxy", type number},
            {"expected_monthly_exposure_proxy", type number}
        },
        "en-US"
    )
in
    FactScoring
