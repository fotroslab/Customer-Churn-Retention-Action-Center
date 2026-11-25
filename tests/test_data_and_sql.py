"""Tests for SQL staging, cleaning, pseudonymization, leakage prevention, and Star Schema integrity."""

from __future__ import annotations

import sqlite3

import pandas as pd

from src.config import POWERBI_MART_DIR, WAREHOUSE_DB_PATH, load_project_config
from src.features import ALL_MODEL_FEATURES


def test_sql_warehouse_tables_and_row_counts() -> None:
    assert WAREHOUSE_DB_PATH.exists(), "SQLite warehouse DB must exist after pipeline execution."
    with sqlite3.connect(WAREHOUSE_DB_PATH) as conn:
        clean_df = pd.read_sql("SELECT * FROM cln_customer_subscription_clean", conn)
        feat_df = pd.read_sql("SELECT * FROM feat_customer_churn_features", conn)
        audit_df = pd.read_sql("SELECT * FROM stg_pipeline_audit_log", conn)

    assert len(clean_df) == 7043
    assert len(feat_df) == 7043
    assert clean_df["synthetic_customer_id"].is_unique
    assert "customerID" not in clean_df.columns
    assert "customerID" not in feat_df.columns
    assert (audit_df["check_status"] == "PASS").all()


def test_zero_tenure_blank_total_charges_handled_deterministically() -> None:
    with sqlite3.connect(WAREHOUSE_DB_PATH) as conn:
        clean_df = pd.read_sql("SELECT * FROM cln_customer_subscription_clean", conn)

    imputed_rows = clean_df[clean_df["was_total_charges_zero_imputed"] == 1]
    assert len(imputed_rows) == 11
    assert (imputed_rows["tenure_months"] == 0).all()
    assert (imputed_rows["total_charges"] == 0.0).all()
    assert clean_df["total_charges"].isna().sum() == 0
    assert (clean_df["monthly_charges"] >= 0).all()


def test_no_target_leakage_in_feature_list() -> None:
    forbidden = {"churn", "churn_label", "churn_status_text", "observed_churn_flag", "churn_probability"}
    overlap = forbidden.intersection({c.lower() for c in ALL_MODEL_FEATURES})
    assert not overlap, f"Target leakage column detected in ALL_MODEL_FEATURES: {overlap}"


def test_star_schema_referential_integrity() -> None:
    dim_cust = pd.read_csv(POWERBI_MART_DIR / "dim_customer.csv")
    dim_cb = pd.read_csv(POWERBI_MART_DIR / "dim_contract_billing.csv")
    dim_sb = pd.read_csv(POWERBI_MART_DIR / "dim_service_bundle.csv")
    dim_rb = pd.read_csv(POWERBI_MART_DIR / "dim_risk_band.csv")
    fact = pd.read_csv(POWERBI_MART_DIR / "fact_customer_churn_scoring.csv")
    queue = pd.read_csv(POWERBI_MART_DIR / "fact_retention_action_queue.csv")

    assert len(fact) == 7043
    assert set(fact["customer_sk"]).issubset(set(dim_cust["customer_sk"]))
    assert set(fact["contract_billing_key"]).issubset(set(dim_cb["contract_billing_key"]))
    assert set(fact["service_bundle_key"]).issubset(set(dim_sb["service_bundle_key"]))
    assert set(fact["risk_band_key"]).issubset(set(dim_rb["risk_band_key"]))
    assert set(queue["synthetic_customer_id"]).issubset(set(dim_cust["synthetic_customer_id"]))


def test_author_config_has_no_fabricated_links() -> None:
    cfg = load_project_config()
    author = cfg["author"]
    assert author["github_url"] == "https://github.com/fotroslab/"
    assert author["email"] == ""
    assert author["linkedin_url"] == ""
