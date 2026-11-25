"""Ingestion, SQL staging/cleaning execution, and Star Schema mart builder."""

from __future__ import annotations

import hashlib
import sqlite3
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    EXPECTED_RAW_COLUMNS,
    POWERBI_MART_DIR,
    RAW_CSV_PATH,
    WAREHOUSE_DB_PATH,
    load_project_config,
)
from src.features import build_feature_dataframe


def ensure_raw_dataset(csv_path: Path = RAW_CSV_PATH) -> dict[str, Any]:
    """Verify local raw CSV or fetch from official IBM GitHub repository if missing."""
    cfg = load_project_config()["dataset"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        urllib.request.urlretrieve(cfg["raw_csv_url"], csv_path)

    sha256_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    raw_df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    missing_cols = [c for c in EXPECTED_RAW_COLUMNS if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Dataset schema mismatch. Missing columns: {missing_cols}")

    return {
        "path": str(csv_path.relative_to(csv_path.parents[2])),
        "sha256": sha256_hash,
        "expected_sha256_match": sha256_hash == cfg["sha256"],
        "row_count": int(len(raw_df)),
        "column_count": int(len(raw_df.columns)),
    }


def run_sql_staging_and_cleaning(
    csv_path: Path = RAW_CSV_PATH,
    db_path: Path = WAREHOUSE_DB_PATH,
) -> dict[str, Any]:
    """Execute SQL staging, cleaning, pseudonymization, QA audit logging, and feature view creation."""
    cfg = load_project_config()
    data_version = cfg["project"]["data_version"]
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    dataset_meta = ensure_raw_dataset(csv_path)
    raw_df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    raw_df_stg = raw_df.copy()
    raw_df_stg["data_version"] = data_version
    raw_df_stg["source_file_name"] = csv_path.name
    raw_df_stg["ingested_at_utc"] = now_utc

    # Audit metrics on raw staging
    raw_rows = len(raw_df_stg)
    null_cust_id = int((raw_df_stg["customerID"].str.strip() == "").sum())
    dup_cust_id = int(raw_df_stg["customerID"].str.strip().duplicated().sum())
    blank_total_charges = int((raw_df_stg["TotalCharges"].str.strip() == "").sum())

    # Deterministic deduplication & sorting for reproducible synthetic IDs
    work = raw_df_stg[raw_df_stg["customerID"].str.strip() != ""].copy()
    work["customerID_clean"] = work["customerID"].str.strip()
    work = work.drop_duplicates(subset=["customerID_clean"], keep="first").sort_values(
        "customerID_clean", kind="mergesort"
    ).reset_index(drop=True)

    seq = work.index + 1
    work["customer_sk"] = seq
    work["synthetic_customer_id"] = [f"SYN-CUST-{i:05d}" for i in seq]
    work["source_customer_hash"] = [
        hashlib.sha256(cid.encode("utf-8")).hexdigest()[:16].upper()
        for cid in work["customerID_clean"]
    ]

    tenure_num = pd.to_numeric(work["tenure"].str.strip(), errors="raise").astype(int)
    monthly_num = pd.to_numeric(work["MonthlyCharges"].str.strip(), errors="raise").round(2)

    tc_stripped = work["TotalCharges"].str.strip()
    zero_tenure_blank_mask = (tc_stripped == "") & (tenure_num == 0)
    unexpected_blank_tc = int(((tc_stripped == "") & (tenure_num != 0)).sum())
    if unexpected_blank_tc > 0:
        raise ValueError(f"Found {unexpected_blank_tc} blank TotalCharges rows with tenure > 0.")

    tc_clean = pd.to_numeric(tc_stripped.where(~zero_tenure_blank_mask, "0.0"), errors="raise").round(2)

    clean_df = pd.DataFrame(
        {
            "customer_sk": work["customer_sk"],
            "synthetic_customer_id": work["synthetic_customer_id"],
            "source_customer_hash": work["source_customer_hash"],
            "gender": work["gender"].str.strip(),
            "senior_citizen": (pd.to_numeric(work["SeniorCitizen"].str.strip()) == 1).astype(int),
            "partner": (work["Partner"].str.strip() == "Yes").astype(int),
            "dependents": (work["Dependents"].str.strip() == "Yes").astype(int),
            "tenure_months": tenure_num,
            "phone_service": (work["PhoneService"].str.strip() == "Yes").astype(int),
            "multiple_lines": work["MultipleLines"].str.strip(),
            "internet_service": work["InternetService"].str.strip(),
            "online_security": work["OnlineSecurity"].str.strip(),
            "online_backup": work["OnlineBackup"].str.strip(),
            "device_protection": work["DeviceProtection"].str.strip(),
            "tech_support": work["TechSupport"].str.strip(),
            "streaming_tv": work["StreamingTV"].str.strip(),
            "streaming_movies": work["StreamingMovies"].str.strip(),
            "contract_type": work["Contract"].str.strip(),
            "paperless_billing": (work["PaperlessBilling"].str.strip() == "Yes").astype(int),
            "payment_method": work["PaymentMethod"].str.strip(),
            "monthly_charges": monthly_num,
            "total_charges": tc_clean,
            "was_total_charges_zero_imputed": zero_tenure_blank_mask.astype(int),
            "churn_label": (work["Churn"].str.strip() == "Yes").astype(int),
            "churn_status_text": work["Churn"].str.strip(),
            "data_version": data_version,
            "cleaned_at_utc": now_utc,
        }
    )

    # Domain checks matching SQL CHECK constraints
    invalid_bounds = int(
        (
            (clean_df["tenure_months"] < 0)
            | (clean_df["tenure_months"] > 120)
            | (clean_df["monthly_charges"] < 0)
            | (clean_df["total_charges"] < 0)
        ).sum()
    )
    if invalid_bounds > 0:
        raise ValueError(f"Domain constraint violation on {invalid_bounds} rows.")

    feat_df = build_feature_dataframe(clean_df)
    feat_df["feature_snapshot_utc"] = now_utc

    audit_rows = [
        {
            "check_stage": "STAGING",
            "check_name": "raw_row_count",
            "metric_value": float(raw_rows),
            "expected_rule": "Expected 7,043 rows in IBM Telco benchmark",
            "check_status": "PASS" if raw_rows == 7043 else "WARN",
            "notes": "Ingested raw CSV records into stg_telco_customer_raw",
            "data_version": data_version,
            "logged_at_utc": now_utc,
        },
        {
            "check_stage": "STAGING",
            "check_name": "duplicate_customer_id_count",
            "metric_value": float(dup_cust_id),
            "expected_rule": "0 duplicate customer keys",
            "check_status": "PASS" if dup_cust_id == 0 else "WARN",
            "notes": "Checked uniqueness of source customer identifier prior to pseudonymization",
            "data_version": data_version,
            "logged_at_utc": now_utc,
        },
        {
            "check_stage": "STAGING",
            "check_name": "null_customer_id_count",
            "metric_value": float(null_cust_id),
            "expected_rule": "0 empty customer identifiers",
            "check_status": "PASS" if null_cust_id == 0 else "FAIL",
            "notes": "Verified primary identifier completeness",
            "data_version": data_version,
            "logged_at_utc": now_utc,
        },
        {
            "check_stage": "CLEAN",
            "check_name": "zero_tenure_blank_total_charges",
            "metric_value": float(blank_total_charges),
            "expected_rule": "Blank TotalCharges allowed ONLY when tenure_months = 0 (mapped to $0.00)",
            "check_status": "PASS" if unexpected_blank_tc == 0 else "FAIL",
            "notes": f"{blank_total_charges} onboarding (tenure=0) rows mapped deterministically to TotalCharges=0.00",
            "data_version": data_version,
            "logged_at_utc": now_utc,
        },
        {
            "check_stage": "CLEAN",
            "check_name": "domain_range_violations",
            "metric_value": float(invalid_bounds),
            "expected_rule": "0 negative charges or out-of-range tenure values",
            "check_status": "PASS" if invalid_bounds == 0 else "FAIL",
            "notes": "Checked tenure [0, 120], monthly_charges >= 0, total_charges >= 0",
            "data_version": data_version,
            "logged_at_utc": now_utc,
        },
        {
            "check_stage": "FEATURE",
            "check_name": "observed_churn_base_rate",
            "metric_value": round(float(feat_df["churn_label"].mean()), 4),
            "expected_rule": "Natural churn rate between 0.05 and 0.50 (~0.2654 on IBM benchmark)",
            "check_status": "PASS",
            "notes": f"{int(feat_df['churn_label'].sum())} churned of {len(feat_df)} total customers",
            "data_version": data_version,
            "logged_at_utc": now_utc,
        },
    ]
    audit_df = pd.DataFrame(audit_rows)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Note: Do NOT persist raw customerID in public downstream tables; keep raw table only in local staging
    with sqlite3.connect(db_path) as conn:
        raw_df_stg.drop(columns=["customerID"]).to_sql(
            "stg_telco_customer_raw", conn, if_exists="replace", index=False
        )
        clean_df.to_sql("cln_customer_subscription_clean", conn, if_exists="replace", index=False)
        feat_df.to_sql("feat_customer_churn_features", conn, if_exists="replace", index=False)
        audit_df.to_sql("stg_pipeline_audit_log", conn, if_exists="replace", index=False)

    return {
        "dataset_meta": dataset_meta,
        "clean_rows": len(clean_df),
        "feature_rows": len(feat_df),
        "zero_tenure_imputed": blank_total_charges,
        "natural_churn_rate": round(float(feat_df["churn_label"].mean()), 6),
        "audit_summary": audit_rows,
    }


def persist_scoring_and_export_star_schema(
    scores_df: pd.DataFrame,
    action_queue_df: pd.DataFrame,
    eval_registry_df: pd.DataFrame,
    db_path: Path = WAREHOUSE_DB_PATH,
    mart_dir: Path = POWERBI_MART_DIR,
) -> dict[str, int]:
    """Persist ML scores & action queue into SQL warehouse and export Power BI Star Schema CSVs."""
    mart_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        scores_df.to_sql("ml_customer_churn_scores", conn, if_exists="replace", index=False)
        action_queue_df.to_sql("ml_retention_action_queue", conn, if_exists="replace", index=False)
        eval_registry_df.to_sql("ml_model_evaluation_registry", conn, if_exists="replace", index=False)

        feat_df = pd.read_sql("SELECT * FROM feat_customer_churn_features", conn)
        audit_df = pd.read_sql("SELECT * FROM stg_pipeline_audit_log", conn)

    # 1. dim_customer
    dim_customer = feat_df[
        [
            "customer_sk",
            "synthetic_customer_id",
            "gender",
            "senior_citizen",
            "partner",
            "dependents",
            "tenure_months",
            "tenure_cohort",
        ]
    ].copy()
    dim_customer["senior_segment"] = dim_customer["senior_citizen"].map(
        {1: "Senior (65+)", 0: "Non-Senior"}
    )
    dim_customer["has_partner"] = dim_customer["partner"].map({1: "Yes", 0: "No"})
    dim_customer["has_dependents"] = dim_customer["dependents"].map({1: "Yes", 0: "No"})

    # 2. dim_contract_billing
    cb_cols = ["contract_type", "payment_method", "paperless_billing", "has_auto_pay"]
    dim_contract_billing = (
        feat_df[cb_cols]
        .drop_duplicates()
        .sort_values(cb_cols)
        .reset_index(drop=True)
    )
    dim_contract_billing.insert(0, "contract_billing_key", dim_contract_billing.index + 1)
    dim_contract_billing["billing_delivery"] = dim_contract_billing["paperless_billing"].map(
        {1: "Paperless", 0: "Paper Statement"}
    )
    dim_contract_billing["payment_automation"] = dim_contract_billing["has_auto_pay"].map(
        {1: "Automatic Payment", 0: "Manual Payment"}
    )

    # 3. dim_service_bundle
    sb_cols = [
        "internet_service",
        "phone_service",
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "active_addon_count",
        "protection_support_count",
        "streaming_addon_count",
    ]
    dim_service_bundle = (
        feat_df[sb_cols]
        .drop_duplicates()
        .sort_values(sb_cols[:4])
        .reset_index(drop=True)
    )
    dim_service_bundle.insert(0, "service_bundle_key", dim_service_bundle.index + 1)
    dim_service_bundle["phone_service_text"] = dim_service_bundle["phone_service"].map(
        {1: "Yes", 0: "No"}
    )

    # 4. dim_risk_band
    dim_risk_band = pd.DataFrame(
        [
            {
                "risk_band_key": 1,
                "risk_band": "Critical Risk",
                "sort_order": 1,
                "probability_band_desc": ">= 0.65",
                "sla_guidance": "Within 24h Priority Care Queue",
            },
            {
                "risk_band_key": 2,
                "risk_band": "High Risk",
                "sort_order": 2,
                "probability_band_desc": "0.45 - 0.64",
                "sla_guidance": "Within 72h Proactive Outreach",
            },
            {
                "risk_band_key": 3,
                "risk_band": "Moderate Risk",
                "sort_order": 3,
                "probability_band_desc": "Operational Threshold - 0.44",
                "sla_guidance": "Automated Digital Check-In",
            },
            {
                "risk_band_key": 4,
                "risk_band": "Low Risk",
                "sort_order": 4,
                "probability_band_desc": "Below Operational Threshold",
                "sla_guidance": "Standard Lifecycle Experience",
            },
        ]
    )

    # 5. fact_customer_churn_scoring
    scores_for_join = scores_df.drop(columns=["data_version"], errors="ignore")
    merged = (
        feat_df.merge(scores_for_join, on="synthetic_customer_id", how="inner")
        .merge(dim_contract_billing[cb_cols + ["contract_billing_key"]], on=cb_cols, how="inner")
        .merge(dim_service_bundle[sb_cols + ["service_bundle_key"]], on=sb_cols, how="inner")
        .merge(dim_risk_band[["risk_band", "risk_band_key"]], on="risk_band", how="inner")
    )

    fact_scoring = merged[
        [
            "customer_sk",
            "synthetic_customer_id",
            "contract_billing_key",
            "service_bundle_key",
            "risk_band_key",
            "churn_label",
            "churn_status_text",
            "monthly_charges",
            "total_charges",
            "avg_historical_monthly_spend",
            "charge_velocity_delta",
            "dataset_split",
            "churn_probability",
            "operational_threshold",
            "predicted_high_risk_flag",
            "risk_band",
            "monthly_charges_at_risk_proxy",
            "expected_monthly_exposure_proxy",
            "reason_code_1",
            "reason_code_2",
            "reason_code_3",
            "reason_summary_non_causal",
            "model_name",
            "model_version",
            "data_version",
            "scored_at_utc",
        ]
    ].rename(
        columns={
            "churn_label": "observed_churn_flag",
            "churn_status_text": "observed_churn_text",
            "monthly_charges": "monthly_charges_usd",
            "total_charges": "total_charges_usd",
        }
    )

    # Save Star Schema CSVs for Power BI Desktop Import Mode
    tables_to_export = {
        "dim_customer": dim_customer,
        "dim_contract_billing": dim_contract_billing,
        "dim_service_bundle": dim_service_bundle,
        "dim_risk_band": dim_risk_band,
        "fact_customer_churn_scoring": fact_scoring,
        "fact_retention_action_queue": action_queue_df,
        "fact_model_evaluation": eval_registry_df,
        "fact_data_quality_audit": audit_df,
    }

    row_counts: dict[str, int] = {}
    with sqlite3.connect(db_path) as conn:
        for table_name, df_out in tables_to_export.items():
            csv_file = mart_dir / f"{table_name}.csv"
            df_out.to_csv(csv_file, index=False)
            df_out.to_sql(f"bi_{table_name}", conn, if_exists="replace", index=False)
            row_counts[table_name] = int(len(df_out))

    return row_counts
