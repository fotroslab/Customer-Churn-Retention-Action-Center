"""Leakage-free row-local feature derivation and strict input validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "active_addon_count",
    "protection_support_count",
    "streaming_addon_count",
    "avg_historical_monthly_spend",
    "charge_velocity_delta",
]

BINARY_FEATURES = [
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "paperless_billing",
    "is_month_to_month",
    "is_fiber_optic",
    "is_electronic_check",
    "has_auto_pay",
]

CATEGORICAL_FEATURES = [
    "gender",
    "tenure_cohort",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract_type",
    "payment_method",
]

ALL_MODEL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

VALID_GENDERS = {"Female", "Male"}
VALID_MULTIPLE_LINES = {"Yes", "No", "No phone service"}
VALID_INTERNET = {"DSL", "Fiber optic", "No"}
VALID_INTERNET_ADDONS = {"Yes", "No", "No internet service"}
VALID_CONTRACTS = {"Month-to-month", "One year", "Two year"}
VALID_PAYMENTS = {
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
}


def assign_tenure_cohort(tenure_months: int | float) -> str:
    """Assign deterministic onboarding/loyalty cohort from observable tenure."""
    t = int(tenure_months)
    if t <= 6:
        return "0-6m Onboarding"
    if t <= 12:
        return "7-12m Early"
    if t <= 24:
        return "13-24m Developing"
    if t <= 48:
        return "25-48m Established"
    return "49m+ Loyal"


def validate_and_normalize_customer_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate single-customer interactive inputs and enforce service consistency rules."""
    errors: list[str] = []

    try:
        tenure = int(record.get("tenure_months", record.get("tenure", 0)))
    except (TypeError, ValueError):
        raise ValueError("Tenure must be a valid integer between 0 and 120 months.")

    if tenure < 0 or tenure > 120:
        errors.append("Tenure must be between 0 and 120 months.")

    try:
        monthly_charges = float(record.get("monthly_charges", record.get("MonthlyCharges", 0.0)))
    except (TypeError, ValueError):
        raise ValueError("Monthly charges must be a valid numeric value in USD.")

    if not np.isfinite(monthly_charges) or monthly_charges < 0.0 or monthly_charges > 500.0:
        errors.append("Monthly charges must be between $0.00 and $500.00.")

    raw_tc = record.get("total_charges", record.get("TotalCharges", None))
    if raw_tc is None or (isinstance(raw_tc, str) and raw_tc.strip() == ""):
        total_charges = 0.0 if tenure == 0 else round(monthly_charges * tenure, 2)
    else:
        try:
            total_charges = float(raw_tc)
        except (TypeError, ValueError):
            raise ValueError("Total charges must be a valid non-negative numeric value.")

    if not np.isfinite(total_charges) or total_charges < 0.0 or total_charges > 100000.0:
        errors.append("Total charges must be between $0.00 and $100,000.00.")

    if tenure == 0:
        total_charges = 0.0

    gender = str(record.get("gender", "Female")).strip()
    if gender not in VALID_GENDERS:
        errors.append(f"Invalid gender '{gender}'. Expected one of {sorted(VALID_GENDERS)}.")

    contract_type = str(record.get("contract_type", record.get("Contract", "Month-to-month"))).strip()
    if contract_type not in VALID_CONTRACTS:
        errors.append(f"Invalid contract '{contract_type}'. Expected one of {sorted(VALID_CONTRACTS)}.")

    payment_method = str(record.get("payment_method", record.get("PaymentMethod", "Electronic check"))).strip()
    if payment_method not in VALID_PAYMENTS:
        errors.append(f"Invalid payment method '{payment_method}'.")

    internet_service = str(record.get("internet_service", record.get("InternetService", "Fiber optic"))).strip()
    if internet_service not in VALID_INTERNET:
        errors.append(f"Invalid internet service '{internet_service}'.")

    phone_raw = record.get("phone_service", record.get("PhoneService", 1))
    phone_service = 1 if str(phone_raw).strip() in {"1", "Yes", "True", "true"} else 0

    multiple_lines = str(record.get("multiple_lines", record.get("MultipleLines", "No"))).strip()
    if phone_service == 0:
        multiple_lines = "No phone service"
    elif multiple_lines == "No phone service":
        multiple_lines = "No"
    if multiple_lines not in VALID_MULTIPLE_LINES:
        errors.append(f"Invalid multiple_lines '{multiple_lines}'.")

    addon_fields = [
        ("online_security", "OnlineSecurity"),
        ("online_backup", "OnlineBackup"),
        ("device_protection", "DeviceProtection"),
        ("tech_support", "TechSupport"),
        ("streaming_tv", "StreamingTV"),
        ("streaming_movies", "StreamingMovies"),
    ]
    normalized_addons: dict[str, str] = {}
    for snake_key, pascal_key in addon_fields:
        val = str(record.get(snake_key, record.get(pascal_key, "No"))).strip()
        if internet_service == "No":
            val = "No internet service"
        elif val == "No internet service":
            val = "No"
        if val not in VALID_INTERNET_ADDONS:
            errors.append(f"Invalid value '{val}' for {snake_key}.")
        normalized_addons[snake_key] = val

    if errors:
        raise ValueError(" | ".join(errors))

    def _to_bit(v: Any) -> int:
        return 1 if str(v).strip() in {"1", "Yes", "True", "true"} else 0

    normalized = {
        "gender": gender,
        "senior_citizen": _to_bit(record.get("senior_citizen", record.get("SeniorCitizen", 0))),
        "partner": _to_bit(record.get("partner", record.get("Partner", 0))),
        "dependents": _to_bit(record.get("dependents", record.get("Dependents", 0))),
        "tenure_months": tenure,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        **normalized_addons,
        "contract_type": contract_type,
        "paperless_billing": _to_bit(record.get("paperless_billing", record.get("PaperlessBilling", 1))),
        "payment_method": payment_method,
        "monthly_charges": round(monthly_charges, 2),
        "total_charges": round(total_charges, 2),
    }
    return normalized


def build_feature_dataframe(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly row-local observable features matching sql/03_feature_engineering_views.sql.

    Zero population statistics (mean/std/quantiles) are computed here so this function
    is 100% free of cross-row or train/test data leakage.
    """
    df = clean_df.copy()
    df["tenure_cohort"] = df["tenure_months"].apply(assign_tenure_cohort)

    addon_cols = [
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]
    prot_cols = [
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
    ]
    stream_cols = ["streaming_tv", "streaming_movies"]

    df["active_addon_count"] = sum((df[col] == "Yes").astype(int) for col in addon_cols)
    df["protection_support_count"] = sum((df[col] == "Yes").astype(int) for col in prot_cols)
    df["streaming_addon_count"] = sum((df[col] == "Yes").astype(int) for col in stream_cols)

    df["is_month_to_month"] = (df["contract_type"] == "Month-to-month").astype(int)
    df["is_fiber_optic"] = (df["internet_service"] == "Fiber optic").astype(int)
    df["is_electronic_check"] = (df["payment_method"] == "Electronic check").astype(int)
    df["has_auto_pay"] = df["payment_method"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)

    avg_hist = np.where(
        df["tenure_months"] > 0,
        df["total_charges"] / np.maximum(df["tenure_months"], 1),
        df["monthly_charges"],
    )
    df["avg_historical_monthly_spend"] = np.round(avg_hist, 2)
    df["charge_velocity_delta"] = np.round(df["monthly_charges"] - df["avg_historical_monthly_spend"], 2)

    return df
