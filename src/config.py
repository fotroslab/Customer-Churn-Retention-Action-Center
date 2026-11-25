"""Central configuration, visual palette, and cloud-safe relative path constants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "project_config.json"

DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_PATH = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
WAREHOUSE_DB_PATH = DATA_DIR / "warehouse" / "churn_intelligence.db"
POWERBI_MART_DIR = DATA_DIR / "powerbi_mart"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_BUNDLE_PATH = MODELS_DIR / "model_bundle.joblib"
METRICS_REPORT_PATH = MODELS_DIR / "metrics_report.json"
MODEL_COMPARISON_CSV_PATH = MODELS_DIR / "model_comparison.csv"
THRESHOLD_CURVE_CSV_PATH = MODELS_DIR / "threshold_curve.csv"
SHAP_GLOBAL_CSV_PATH = MODELS_DIR / "shap_global_summary.csv"
CALIBRATION_CURVE_CSV_PATH = MODELS_DIR / "calibration_curve.csv"

SQL_DIR = PROJECT_ROOT / "sql"
POWERBI_DIR = PROJECT_ROOT / "powerbi"
SNAPSHOTS_DIR = POWERBI_DIR / "snapshots"

# Visual Design System (Master Portfolio Standard)
PALETTE = {
    "bg": "#0B1220",
    "panel": "#141F30",
    "primary": "#5EE0B5",
    "secondary": "#8AB4F8",
    "text_main": "#F1F5F9",
    "text_muted": "#ACBBD0",
    "border": "#304158",
    "critical": "#F87171",
    "high": "#FBBF24",
    "moderate": "#8AB4F8",
    "low": "#5EE0B5",
}

EXPECTED_RAW_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def load_project_config() -> dict[str, Any]:
    """Load project metadata and operational assumptions from JSON config."""
    with CONFIG_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)
