# Power BI Desktop (`.pbix`) Build, SQL Server Connection & Refresh Guide

> **Technical Transparency Note:** Binary `.pbix` files are produced by Microsoft's proprietary VertiPaq engine inside the Windows-only **Power BI Desktop** application and cannot be natively compiled in a Linux cloud environment without Power BI Desktop. Following professional portfolio standards, no non-native file with a fake `.pbix` extension has been created; instead, **all ready-to-assemble components (roughly 10 to 15 minutes to build)** have been generated and tested in this folder.

---

## 1. Two Data Loading Methods in Power BI Desktop

### Method A (Fastest Path — No SQL Server Installation Required): Connect to the Ready-Made Star Schema Mart
Running `python scripts/run_pipeline.py` produces the eight star-schema tables in `data/powerbi_mart/`:
1. On Windows, open **Power BI Desktop** (free download from the Microsoft Store).
2. From the **Home → Get Data → Text/CSV** tab, load the eight files below from the `data/powerbi_mart/` folder (or use the ready-made script `powerbi/powerquery/load_from_csv_mart.pq` with the `pMartFolderPath` parameter):
   - `dim_customer.csv` (7,043 rows)
   - `dim_contract_billing.csv` (24 rows)
   - `dim_service_bundle.csv` (322 rows)
   - `dim_risk_band.csv` (4 rows)
   - `fact_customer_churn_scoring.csv` (7,043 rows)
   - `fact_retention_action_queue.csv` (2,020 rows)
   - `fact_model_evaluation.csv` (18 rows)
   - `fact_data_quality_audit.csv` (6 rows)

### Method B (Enterprise SQL Server Express / Developer Path):
1. In **SSMS** or **Azure Data Studio**, run the scripts in the `sql/` folder in order:
   - `sql/01_create_schemas_and_staging.sql` (creates database `ChurnIntelligenceDB` and schemas `stg`, `cln`, `feat`, `ml`, `bi`)
   - Load the file `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` with the `BULK INSERT` command into `stg.telco_customer_raw` (or the Flat File Import wizard in SSMS)
   - `sql/02_clean_and_validate.sql`
   - `sql/03_feature_engineering_views.sql`
   - `sql/04_scoring_and_action_tables.sql`
   - Load the Python scoring outputs (`data/powerbi_mart/fact_customer_churn_scoring.csv` and `fact_retention_action_queue.csv`) or connect directly via SQLAlchemy
   - `sql/05_powerbi_star_schema_views.sql`
2. In Power BI Desktop, choose **Get Data → SQL Server**, enter the server address (e.g., `localhost\SQLEXPRESS`) and database `ChurnIntelligenceDB`, and select the `bi` schema views in **Import** mode (the ready-made Power Query code is in `powerbi/powerquery/load_from_sql_server.pq`).

---

## 2. Applying the Visual Theme (`#0B1220`)
1. In Power BI Desktop, go to the **View → Themes → Browse for themes** tab.
2. Select the file `powerbi/theme/ChurnIntelligenceDarkTheme.json` so the `#0B1220` background, `#141F30` cards, `#5EE0B5` primary color, and `#8AB4F8` secondary color are applied automatically to all pages and visuals.

---

## 3. Configuring Star Schema Model Relationships (Model View)
Following the `powerbi/MODEL_SCHEMA_DIAGRAM.svg` diagram, in the **Model View** create many-to-one (`1:*`) relationships with **Single** filter direction from the dimension tables to the fact table:
- `dim_customer[customer_sk]` (1) ──► (*) `fact_customer_churn_scoring[customer_sk]`
- `dim_contract_billing[contract_billing_key]` (1) ──► (*) `fact_customer_churn_scoring[contract_billing_key]`
- `dim_service_bundle[service_bundle_key]` (1) ──► (*) `fact_customer_churn_scoring[service_bundle_key]`
- `dim_risk_band[risk_band_key]` (1) ──► (*) `fact_customer_churn_scoring[risk_band_key]`
- `dim_customer[synthetic_customer_id]` (1) ──► (0..1 / *) `fact_retention_action_queue[synthetic_customer_id]`
- Click the `dim_risk_band[risk_band]` column and select the `sort_order` column from the **Sort by column** menu so the order `Critical → High → Moderate → Low` is always preserved.

---

## 4. Adding All DAX Measures in One Place
1. In newer versions of Power BI Desktop, enter the **DAX Query View** (left side of the page).
2. Paste the full content of the file `powerbi/dax/measures_library.dax`.
3. Click the **Update model with changes** button so all 25 DAX measures are added to the model automatically, then set percentage formatting (`0.0%`) and currency formatting (`$#,0`) on the relevant measures.

---

## 5. Standard 4-Page Report Layout (matching `powerbi/snapshots/`)

| Page # | Page Name | Reference Snapshot | Key Components and Filters |
|---|---|---|---|
| **Page 1** | `Churn Summary` | `page1_churn_summary.png` | 5 KPI cards in the top strip + bar chart of churn rate by `contract_type` + grouped chart of observed churn vs. calibrated risk across `tenure_cohort` + churn rate by `internet_service` + probability-weighted financial exposure by `payment_method`. Slicers: `contract_type`, `internet_service`, `senior_segment`. |
| **Page 2** | `Churn Risk Register` | `page2_churn_risk_register.png` | Cards of customer counts per `risk_band` + global associated-factor chart (`models/shap_global_summary.csv`) + gross monthly financial exposure per risk band + table of customers sorted by `churn_probability` and `monthly_charges_usd`, showing `synthetic_customer_id` and non-causal reason codes. |
| **Page 3** | `Recovery Action Center` | `page3_recovery_action_center.png` | Action queue cards (`P1`, `P2`, `P3`) + distribution of analytical playbook recommendations (`recommended_playbook`) + specialist team workload allocation (`assigned_team`) + operational tracking table with slicers `priority_tier`, `assigned_team`, `outreach_status` and an explicit caption "analytical hypothesis for A/B testing — not proven causal effect". |
| **Page 4** | `Model & Data Quality` | `page4_model_and_data_quality.png` | Cards for `Champion Test ROC-AUC (0.8478)`, `PR-AUC (0.6640)`, `Brier (0.1351)` and the status of the 6 SQL quality checks + imbalance-strategy comparison chart (`Natural` vs. `Class-Weight` and `In-Fold SMOTE`) + calibration curve (`models/calibration_curve.csv`) + the `fact_data_quality_audit` table. |

---

## 6. Data Refresh and Safe Publishing Instructions
- **Local Refresh:** After each run of `python scripts/run_pipeline.py`, in Power BI Desktop simply click the **Refresh** button so the tables are re-read from the `data/powerbi_mart/` folder or the `ChurnIntelligenceDB` database.
- **Saving the Final File:** Save the report as `Customer_Churn_Intelligence_Action_Center.pbix` in the `powerbi/` folder.
- **Publish-to-Web Security Warning:** The `Publish to web (public)` capability in Power BI Service requires a Microsoft Fabric work/school account (with tenant admin enablement) and places the report and its underlying data **completely and unauthenticated on the public internet**. Never publish real or personal customer data with `Publish to web`; this capability is only permitted for public/synthetic portfolio datasets.
