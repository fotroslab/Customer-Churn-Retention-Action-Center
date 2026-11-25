"""SHAP explainability, OHE-additive feature aggregation, and non-causal playbooks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

FEATURE_DISPLAY_NAMES = {
    "contract_type": "Contract Commitment Structure",
    "tenure_months": "Subscription Tenure (Months)",
    "tenure_cohort": "Lifecycle Tenure Cohort",
    "internet_service": "Internet Access Tier",
    "monthly_charges": "Current Monthly Bill (USD)",
    "total_charges": "Cumulative Historical Spend (USD)",
    "payment_method": "Billing Payment Method",
    "online_security": "Online Security Add-On",
    "tech_support": "Technical Support Add-On",
    "protection_support_count": "Protection & Support Bundle Depth",
    "active_addon_count": "Total Active Add-On Count",
    "streaming_addon_count": "Streaming Entertainment Add-Ons",
    "paperless_billing": "Paperless Statement Preference",
    "multiple_lines": "Multiple Phone Lines",
    "online_backup": "Online Backup Add-On",
    "device_protection": "Device Protection Add-On",
    "streaming_tv": "Streaming TV Service",
    "streaming_movies": "Streaming Movies Service",
    "senior_citizen": "Senior Citizen Segment",
    "partner": "Partner Household Status",
    "dependents": "Dependents in Household",
    "phone_service": "Voice Phone Service",
    "gender": "Recorded Gender",
    "is_month_to_month": "Month-to-Month Flexibility Flag",
    "is_fiber_optic": "Fiber Optic Tier Flag",
    "is_electronic_check": "Manual Electronic Check Flag",
    "has_auto_pay": "Automatic Payment Enrollment",
    "avg_historical_monthly_spend": "Historical Avg Monthly Spend",
    "charge_velocity_delta": "Recent Bill vs Historical Avg Delta",
}


def map_feature_to_reason_code(parent_feature: str, customer_row: pd.Series) -> tuple[str, str]:
    """Map an associated risk feature and observed customer value to a non-causal reason code."""
    val = customer_row.get(parent_feature)

    if parent_feature in {"contract_type", "is_month_to_month"}:
        if customer_row.get("contract_type") == "Month-to-month":
            return (
                "RC_CONTRACT_MTM",
                "Month-to-month contract term (historically associated with higher exit flexibility)",
            )
        return ("RC_CONTRACT_TERM", f"Contract structure ({customer_row.get('contract_type')})")

    if parent_feature in {"tenure_months", "tenure_cohort", "total_charges"}:
        t = int(customer_row.get("tenure_months", 0))
        if t <= 6:
            return (
                "RC_ONBOARDING_EARLY",
                f"Early onboarding window ({t} mo tenure; elevated early-lifecycle attrition pattern)",
            )
        if t <= 18:
            return (
                "RC_TENURE_DEVELOPING",
                f"Developing lifecycle stage ({t} mo tenure; pre-loyalty stabilization phase)",
            )
        return ("RC_TENURE_PROFILE", f"Tenure & cumulative spend profile ({t} mo)")

    if parent_feature in {"internet_service", "is_fiber_optic"}:
        if customer_row.get("internet_service") == "Fiber optic":
            return (
                "RC_FIBER_EXPOSURE",
                "Fiber optic tier (correlates with higher price sensitivity & competitive switching)",
            )
        return ("RC_INTERNET_TIER", f"Internet service tier ({customer_row.get('internet_service')})")

    if parent_feature in {"online_security", "tech_support", "protection_support_count", "online_backup", "device_protection"}:
        prot_cnt = int(customer_row.get("protection_support_count", 0))
        if prot_cnt == 0:
            return (
                "RC_NO_SUPPORT_SHIELD",
                "Zero active Security/Tech Support add-ons (associated with lower service stickiness)",
            )
        return (
            "RC_THIN_SUPPORT_BUNDLE",
            f"Limited support/protection bundle depth ({prot_cnt} of 4 services active)",
        )

    if parent_feature in {"payment_method", "is_electronic_check", "has_auto_pay"}:
        if customer_row.get("payment_method") == "Electronic check":
            return (
                "RC_MANUAL_ECHECK",
                "Manual electronic check billing (associated with higher monthly payment friction)",
            )
        return ("RC_PAYMENT_CHANNEL", f"Payment channel ({customer_row.get('payment_method')})")

    if parent_feature in {"monthly_charges", "charge_velocity_delta", "avg_historical_monthly_spend"}:
        mc = float(customer_row.get("monthly_charges", 0.0))
        delta = float(customer_row.get("charge_velocity_delta", 0.0))
        if delta > 3.0:
            return (
                "RC_BILL_ESCALATION",
                f"Current bill (${mc:.2f}/mo) exceeds historical monthly average (+${delta:.2f}/mo)",
            )
        return (
            "RC_HIGH_MONTHLY_BILL",
            f"Monthly recurring bill level (${mc:.2f}/mo gross exposure)",
        )

    if parent_feature == "paperless_billing" and int(customer_row.get("paperless_billing", 0)) == 1:
        return ("RC_PAPERLESS_DIGITAL", "Digital/paperless billing profile")

    if parent_feature in {"streaming_tv", "streaming_movies", "streaming_addon_count"}:
        return ("RC_STREAMING_BUNDLE", "Streaming add-on cost/bundle profile")

    if parent_feature == "multiple_lines":
        return ("RC_MULTI_LINE_PROFILE", f"Multi-line voice configuration ({val})")

    if parent_feature == "senior_citizen" and int(customer_row.get("senior_citizen", 0)) == 1:
        return ("RC_SENIOR_SEGMENT", "Senior subscriber segment (higher observed benchmark churn)")

    label = FEATURE_DISPLAY_NAMES.get(parent_feature, parent_feature)
    return (f"RC_{parent_feature.upper()[:16]}", f"Associated pattern in {label} ({val})")


def recommend_retention_playbook(
    reason_codes: str | list[str],
    risk_band: str,
    customer_row: pd.Series,
) -> tuple[str, str, str]:
    """Return analytical rule-based outreach suggestion, non-causal note, and suggested care team.

    Uses the combination of top SHAP risk signals and lifecycle stage so month-to-month
    subscribers are segmented into actionable operational queues rather than a single bucket.
    Note: Playbooks are operational hypotheses for controlled A/B testing, NOT proven causal cures.
    """
    if risk_band == "Low Risk":
        return (
            "No Proactive Call — Standard Lifecycle",
            "Below operational threshold; avoid unnecessary outreach cost or churn sensitization.",
            "Automated Lifecycle",
        )

    codes = [reason_codes] if isinstance(reason_codes, str) else list(reason_codes)
    code_set = set(codes)
    tenure = int(customer_row.get("tenure_months", 12))
    mc = float(customer_row.get("monthly_charges", 0.0))

    # 1. Early onboarding subscribers (0-6 months) need setup/adoption care before pitch-heavy contract locks
    if "RC_ONBOARDING_EARLY" in code_set or tenure <= 6:
        return (
            "90-Day Onboarding Concierge & Setup Check",
            "Rule-based hypothesis: resolve early setup/service friction during initial 6-month onboarding window.",
            "Onboarding Care Desk",
        )

    # 2. High-bill Fiber subscribers without support shield
    if ("RC_FIBER_EXPOSURE" in code_set or customer_row.get("internet_service") == "Fiber optic") and mc >= 85.0:
        return (
            "Fiber Quality Audit & Tier Right-Sizing",
            "Rule-based hypothesis: inspect line performance and review bundle value for high-bill Fiber accounts.",
            "VIP & Fiber Retention Desk",
        )

    # 3. Subscribers lacking Online Security / Tech Support shield
    if "RC_NO_SUPPORT_SHIELD" in code_set or "RC_THIN_SUPPORT_BUNDLE" in code_set:
        return (
            "Security & Tech Support Value Bundle Trial",
            "Rule-based hypothesis: test complimentary support/security trial to evaluate service stickiness uplift.",
            "Customer Success Care",
        )

    # 4. Manual electronic check friction
    if "RC_MANUAL_ECHECK" in code_set:
        return (
            "Auto-Pay Convenience & Billing Check-In",
            "Rule-based hypothesis: reduce monthly manual payment friction via optional automatic billing setup.",
            "Billing Experience Desk",
        )

    # 5. Established month-to-month subscribers eligible for annual value review
    if "RC_CONTRACT_MTM" in code_set:
        return (
            "Annual Plan Rate-Lock Value Review",
            "Rule-based hypothesis: compare 12-month commitment savings vs month-to-month rate; test via A/B holdout.",
            "Retention Specialist Desk",
        )

    return (
        "Proactive Account Health Discovery Call",
        "Rule-based hypothesis: structured care check-in; observational risk signals only.",
        "Customer Success Care",
    )


def aggregate_transformed_shap_to_parent_features(
    raw_shap_matrix: np.ndarray,
    transformed_feature_names: list[str],
    parent_features: list[str],
) -> pd.DataFrame:
    """Sum one-hot / scaled SHAP values back to exact parent input features via Shapley additivity."""
    agg = {pf: np.zeros(raw_shap_matrix.shape[0], dtype=float) for pf in parent_features}

    # Sort parent features longest-first so prefix matching is unambiguous
    sorted_parents = sorted(parent_features, key=len, reverse=True)

    for col_idx, t_name in enumerate(transformed_feature_names):
        clean_name = t_name.split("__", 1)[-1] if "__" in t_name else t_name
        matched_parent = None
        for pf in sorted_parents:
            if clean_name == pf or clean_name.startswith(f"{pf}_"):
                matched_parent = pf
                break
        if matched_parent is not None:
            agg[matched_parent] += raw_shap_matrix[:, col_idx]

    return pd.DataFrame(agg)


def compute_shap_explanations(
    preprocessor: Any,
    base_estimator: Any,
    X_df: pd.DataFrame,
    parent_features: list[str],
    background_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-customer parent-feature SHAP matrix and global SHAP importance table."""
    X_trans = preprocessor.transform(X_df[parent_features])
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    X_trans = np.asarray(X_trans, dtype=float)

    t_names = list(preprocessor.get_feature_names_out())

    import warnings
    est_cls_name = type(base_estimator).__name__.lower()
    if "lgbm" in est_cls_name or "xgb" in est_cls_name or "forest" in est_cls_name:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            explainer = shap.TreeExplainer(base_estimator)
            sv = explainer.shap_values(X_trans)
        if isinstance(sv, list):
            sv_pos = np.asarray(sv[1], dtype=float)
        elif sv.ndim == 3:
            sv_pos = np.asarray(sv[:, :, 1], dtype=float)
        else:
            sv_pos = np.asarray(sv, dtype=float)
    else:
        bg_source = background_df if background_df is not None else X_df
        bg_trans = preprocessor.transform(bg_source[parent_features].iloc[:300])
        if hasattr(bg_trans, "toarray"):
            bg_trans = bg_trans.toarray()
        bg_trans = np.asarray(bg_trans, dtype=float)
        explainer = shap.LinearExplainer(base_estimator, bg_trans)
        sv_pos = np.asarray(explainer.shap_values(X_trans), dtype=float)

    parent_shap_df = aggregate_transformed_shap_to_parent_features(
        sv_pos, t_names, parent_features
    )
    parent_shap_df.index = X_df.index

    # Consolidate heavily collinear binary indicator aliases into main business concepts for cleaner reason ranking
    consolidated = pd.DataFrame(index=X_df.index)
    concept_groups = {
        "contract_type": ["contract_type", "is_month_to_month"],
        "tenure_months": ["tenure_months", "tenure_cohort"],
        "internet_service": ["internet_service", "is_fiber_optic"],
        "monthly_charges": ["monthly_charges", "avg_historical_monthly_spend", "charge_velocity_delta"],
        "total_charges": ["total_charges"],
        "payment_method": ["payment_method", "is_electronic_check", "has_auto_pay"],
        "protection_support_count": [
            "protection_support_count",
            "online_security",
            "tech_support",
            "online_backup",
            "device_protection",
            "active_addon_count",
        ],
        "streaming_addon_count": ["streaming_addon_count", "streaming_tv", "streaming_movies"],
        "paperless_billing": ["paperless_billing"],
        "multiple_lines": ["multiple_lines", "phone_service"],
        "senior_citizen": ["senior_citizen"],
        "partner": ["partner", "dependents"],
        "gender": ["gender"],
    }

    for concept, cols in concept_groups.items():
        present = [c for c in cols if c in parent_shap_df.columns]
        consolidated[concept] = parent_shap_df[present].sum(axis=1)

    global_rows = []
    for concept in consolidated.columns:
        vals = consolidated[concept].to_numpy()
        global_rows.append(
            {
                "feature_key": concept,
                "feature_display_name": FEATURE_DISPLAY_NAMES.get(concept, concept),
                "mean_abs_shap": round(float(np.mean(np.abs(vals))), 5),
                "mean_positive_shap": round(float(np.mean(np.maximum(vals, 0.0))), 5),
                "pct_customers_pushing_risk_up": round(float(np.mean(vals > 0.01)), 4),
            }
        )
    global_df = (
        pd.DataFrame(global_rows)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return consolidated, global_df
