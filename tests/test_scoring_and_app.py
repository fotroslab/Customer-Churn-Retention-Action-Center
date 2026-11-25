"""Tests for live customer scoring, input validation edge cases, non-causal language, and exports."""

from __future__ import annotations

import json
import sqlite3

import joblib
import pandas as pd
import pytest

from src.config import METRICS_REPORT_PATH, MODEL_BUNDLE_PATH, WAREHOUSE_DB_PATH
from src.export_utils import (
    action_pack_to_excel_bytes,
    dataframe_to_csv_bytes,
    generate_executive_pdf_brief,
)
from src.features import validate_and_normalize_customer_record
from src.modeling import score_single_customer_live


@pytest.fixture(scope="module")
def model_bundle():
    return joblib.load(MODEL_BUNDLE_PATH)


def test_zero_tenure_and_no_internet_normalization(model_bundle) -> None:
    payload = {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": 1,
        "dependents": 1,
        "tenure_months": 0,
        "phone_service": 0,
        "multiple_lines": "Yes",  # Should auto-normalize to 'No phone service'
        "internet_service": "No",
        "online_security": "Yes",  # Should auto-normalize to 'No internet service'
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract_type": "Two year",
        "paperless_billing": 0,
        "payment_method": "Bank transfer (automatic)",
        "monthly_charges": 20.15,
        "total_charges": "",  # Blank string at tenure=0 must normalize to 0.0
    }
    res = score_single_customer_live(payload, bundle=model_bundle)
    assert res["normalized_input"]["total_charges"] == 0.0
    assert res["normalized_input"]["multiple_lines"] == "No phone service"
    assert res["normalized_input"]["online_security"] == "No internet service"
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert len(res["shap_drivers"]) >= 8


@pytest.mark.parametrize(
    "bad_override",
    [
        {"tenure_months": -3},
        {"tenure_months": 250},
        {"monthly_charges": -15.0},
        {"monthly_charges": "not-a-number"},
        {"contract_type": "Lifetime"},
        {"payment_method": "Bitcoin"},
    ],
)
def test_invalid_inputs_rejected_cleanly(bad_override: dict) -> None:
    base = {
        "gender": "Male",
        "senior_citizen": 0,
        "partner": 0,
        "dependents": 0,
        "tenure_months": 10,
        "phone_service": 1,
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract_type": "Month-to-month",
        "paperless_billing": 1,
        "payment_method": "Electronic check",
        "monthly_charges": 75.0,
        "total_charges": 750.0,
    }
    base.update(bad_override)
    with pytest.raises(ValueError):
        validate_and_normalize_customer_record(base)


def test_non_causal_governance_wording_in_action_queue() -> None:
    with sqlite3.connect(WAREHOUSE_DB_PATH) as conn:
        q_df = pd.read_sql("SELECT * FROM bi_fact_retention_action_queue", conn)

    combined_text = " ".join(
        q_df["reason_summary_non_causal"].head(100).tolist()
        + q_df["playbook_analytical_note"].head(100).tolist()
    ).lower()
    assert "proven root cause" not in combined_text
    assert "guaranteed saved revenue" not in combined_text
    assert "hypothesis" in combined_text or "observational" in combined_text


def test_downloadable_exports_csv_xlsx_pdf() -> None:
    with METRICS_REPORT_PATH.open("r", encoding="utf-8") as fp:
        metrics = json.load(fp)
    with sqlite3.connect(WAREHOUSE_DB_PATH) as conn:
        q_df = pd.read_sql("SELECT * FROM bi_fact_retention_action_queue LIMIT 25", conn)
        ev_df = pd.read_sql("SELECT * FROM bi_fact_model_evaluation", conn)
        au_df = pd.read_sql("SELECT * FROM bi_fact_data_quality_audit", conn)

    csv_b = dataframe_to_csv_bytes(q_df)
    assert csv_b.startswith(b"priority_rank,")

    xlsx_b = action_pack_to_excel_bytes(q_df, ev_df, au_df)
    assert xlsx_b is not None and xlsx_b[:2] == b"PK"

    pdf_b = generate_executive_pdf_brief(metrics, q_df)
    assert pdf_b is not None and pdf_b.startswith(b"%PDF-")


def test_streamlit_ui_all_six_views_and_empty_slicer() -> None:
    from streamlit.testing.v1 import AppTest
    from src.config import PROJECT_ROOT

    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=30)
    at.run()
    assert not at.exception

    # Toggle Persian bilingual guide and step through all 6 workspace views
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception

    for opt in list(at.sidebar.radio[0].options):
        at.sidebar.radio[0].set_value(opt).run()
        assert not at.exception, f"Unhandled UI exception on view {opt}: {at.exception}"

    # Verify empty-filter boundary state renders warning cleanly without crashing
    at.sidebar.radio[0].set_value("1. Churn Summary (BI Page 1)").run()
    at.sidebar.multiselect[0].set_value([]).run()
    assert not at.exception
    assert len(at.warning) >= 1

