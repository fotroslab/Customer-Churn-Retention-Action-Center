"""Customer Churn Intelligence & Retention Action Center — Interactive Portfolio Application."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    CALIBRATION_CURVE_CSV_PATH,
    METRICS_REPORT_PATH,
    MODEL_BUNDLE_PATH,
    MODEL_COMPARISON_CSV_PATH,
    PALETTE,
    POWERBI_DIR,
    SHAP_GLOBAL_CSV_PATH,
    SNAPSHOTS_DIR,
    SQL_DIR,
    THRESHOLD_CURVE_CSV_PATH,
    WAREHOUSE_DB_PATH,
    load_project_config,
)
from src.export_utils import (
    action_pack_to_excel_bytes,
    dataframe_to_csv_bytes,
    generate_executive_pdf_brief,
)
from src.modeling import optimize_operational_threshold, score_single_customer_live

st.set_page_config(
    page_title="Customer Churn Intelligence & Retention Action Center",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_design_system_css() -> None:
    """Apply strict Master Portfolio dark visual design system (#0B1220 / #141F30 / #5EE0B5)."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {PALETTE['bg']};
            color: {PALETTE['text_main']};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {PALETTE['panel']};
            border-right: 1px solid {PALETTE['border']};
        }}
        /* Visible keyboard focus ring for accessibility */
        button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible, a:focus-visible {{
            outline: 2px solid {PALETTE['primary']} !important;
            outline-offset: 2px !important;
        }}
        .hero-banner {{
            background: {PALETTE['panel']};
            border: 1px solid {PALETTE['border']};
            border-left: 4px solid {PALETTE['primary']};
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.1rem;
        }}
        .hero-title {{
            font-size: 1.55rem;
            font-weight: 700;
            color: {PALETTE['text_main']};
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.01em;
        }}
        .hero-sub {{
            font-size: 0.95rem;
            color: {PALETTE['text_muted']};
            line-height: 1.55;
            margin-bottom: 0.7rem;
        }}
        .pill-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.4rem;
        }}
        .tech-pill {{
            background: {PALETTE['bg']};
            border: 1px solid {PALETTE['border']};
            color: {PALETTE['secondary']};
            font-size: 0.76rem;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
            gap: 0.85rem;
            margin-bottom: 1.1rem;
        }}
        .kpi-card {{
            background: {PALETTE['panel']};
            border: 1px solid {PALETTE['border']};
            border-radius: 8px;
            padding: 0.95rem 1.05rem;
        }}
        .kpi-label {{
            font-size: 0.73rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {PALETTE['text_muted']};
            font-weight: 600;
            margin-bottom: 0.3rem;
        }}
        .kpi-value {{
            font-size: 1.45rem;
            font-weight: 700;
            color: {PALETTE['primary']};
            line-height: 1.2;
        }}
        .kpi-sub {{
            font-size: 0.76rem;
            color: {PALETTE['secondary']};
            margin-top: 0.28rem;
        }}
        .governance-box {{
            background: rgba(20, 31, 48, 0.85);
            border: 1px solid {PALETTE['border']};
            border-left: 3px solid {PALETTE['secondary']};
            border-radius: 8px;
            padding: 0.75rem 1rem;
            font-size: 0.84rem;
            color: {PALETTE['text_muted']};
            margin-bottom: 1rem;
        }}
        .badge-crit {{ color: #FCA5A5; border: 1px solid #F87171; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.76rem; }}
        .badge-high {{ color: #FDE68A; border: 1px solid #FBBF24; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.76rem; }}
        .badge-mod  {{ color: #BFDBFE; border: 1px solid #8AB4F8; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.76rem; }}
        .badge-low  {{ color: #A7F3D0; border: 1px solid #5EE0B5; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.76rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_artifacts_ready() -> None:
    """Ensure warehouse and model artifacts exist AND are loadable in this environment.

    If the shipped bundle was pickled under a different numpy/sklearn major version
    (the classic cause of ``AttributeError: Can't get attribute ... on ...`` when
    unpickling), it will not load here. Rather than crash, we rebuild the full
    artifact set in-container with the running library versions, so the deployed
    app self-heals without requiring a redeploy for every environment drift.
    """
    if WAREHOUSE_DB_PATH.exists() and MODEL_BUNDLE_PATH.exists():
        # Fast path: shipped artifacts exist. Verify they actually load.
        try:
            joblib.load(MODEL_BUNDLE_PATH)
            return
        except Exception:
            # Incompatible pickle (e.g. numpy 1.x vs 2.x namespaces): rebuild below.
            pass

    with st.spinner(
        "Initializing SQL staging warehouse, calibrated LightGBM model, and SHAP artifacts..."
    ):
        from scripts.run_pipeline import main as run_full_pipeline

        run_full_pipeline()


@st.cache_resource(show_spinner=False)
def load_cached_model_bundle() -> dict[str, Any]:
    """Load the champion model bundle, self-healing if the pickle is incompatible.

    A model pickled with one numpy major version may not load under another
    (e.g. numpy 2.x ``numpy._core`` vs 1.x ``numpy.core`` namespace references).
    On any load failure we rebuild the full artifact set in-container with the
    running library versions and retry, so the deployed app recovers without a
    redeploy for every environment drift.
    """
    try:
        return joblib.load(MODEL_BUNDLE_PATH)
    except Exception as exc:  # noqa: BLE001 - deliberate broad fallback for deployment robustness
        st.warning(
            f"Pre-built model bundle could not be loaded in this environment "
            f"({type(exc).__name__}: {exc}). Rebuilding model and data artifacts "
            "in-container and retrying..."
        )
        from scripts.run_pipeline import main as run_full_pipeline

        run_full_pipeline()
        try:
            return joblib.load(MODEL_BUNDLE_PATH)
        except Exception as exc2:  # noqa: BLE001 - surface a clear failure instead of a raw traceback
            st.error(
                "The model bundle could not be rebuilt automatically in this "
                f"environment ({type(exc2).__name__}: {exc2}). Please trigger a "
                "fresh rebuild of the app (Settings → Dependencies → Rebuild)."
            )
            raise


@st.cache_data(show_spinner=False)
def load_static_artifacts() -> dict[str, Any]:
    cfg = load_project_config()
    with METRICS_REPORT_PATH.open("r", encoding="utf-8") as fp:
        metrics = json.load(fp)
    comp_df = pd.read_csv(MODEL_COMPARISON_CSV_PATH)
    thresh_df = pd.read_csv(THRESHOLD_CURVE_CSV_PATH)
    shap_df = pd.read_csv(SHAP_GLOBAL_CSV_PATH)
    cal_df = pd.read_csv(CALIBRATION_CURVE_CSV_PATH)
    return {
        "config": cfg,
        "metrics": metrics,
        "comparison": comp_df,
        "threshold_curve": thresh_df,
        "shap_global": shap_df,
        "calibration_curve": cal_df,
    }


@st.cache_data(show_spinner=False)
def load_star_schema_joined() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(WAREHOUSE_DB_PATH) as conn:
        fact_joined = pd.read_sql(
            """
            SELECT
                f.customer_sk,
                f.synthetic_customer_id,
                f.dataset_split,
                f.observed_churn_flag,
                f.observed_churn_text,
                f.churn_probability,
                f.operational_threshold,
                f.predicted_high_risk_flag,
                f.risk_band,
                f.monthly_charges_usd,
                f.total_charges_usd,
                f.monthly_charges_at_risk_proxy,
                f.expected_monthly_exposure_proxy,
                f.reason_code_1,
                f.reason_code_2,
                f.reason_code_3,
                f.reason_summary_non_causal,
                c.gender,
                c.senior_segment,
                c.has_partner,
                c.has_dependents,
                c.tenure_months,
                c.tenure_cohort,
                cb.contract_type,
                cb.payment_method,
                cb.billing_delivery,
                cb.payment_automation,
                sb.internet_service,
                sb.phone_service_text,
                sb.online_security,
                sb.tech_support,
                sb.active_addon_count,
                sb.protection_support_count
            FROM bi_fact_customer_churn_scoring f
            JOIN bi_dim_customer c ON f.customer_sk = c.customer_sk
            JOIN bi_dim_contract_billing cb ON f.contract_billing_key = cb.contract_billing_key
            JOIN bi_dim_service_bundle sb ON f.service_bundle_key = sb.service_bundle_key
            """,
            conn,
        )
        eval_df = pd.read_sql("SELECT * FROM bi_fact_model_evaluation", conn)
        audit_df = pd.read_sql("SELECT * FROM bi_fact_data_quality_audit", conn)
    return fact_joined, eval_df, audit_df


def load_live_action_queue() -> pd.DataFrame:
    """Read mutable retention action queue directly so agent follow-up updates reflect immediately."""
    with sqlite3.connect(WAREHOUSE_DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM bi_fact_retention_action_queue ORDER BY priority_rank ASC", conn)


def update_action_queue_record(
    synthetic_customer_id: str,
    outreach_status: str,
    contact_outcome: str,
    agent_notes: str,
) -> None:
    """Persist care specialist follow-up status and outcome in SQLite action table."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with sqlite3.connect(WAREHOUSE_DB_PATH) as conn:
        conn.execute(
            """
            UPDATE bi_fact_retention_action_queue
               SET outreach_status = ?,
                   contact_outcome = ?,
                   agent_notes = ?,
                   last_updated_utc = ?
             WHERE synthetic_customer_id = ?
            """,
            (outreach_status, contact_outcome, agent_notes.strip()[:450], now_utc, synthetic_customer_id),
        )
        conn.execute(
            """
            UPDATE ml_retention_action_queue
               SET outreach_status = ?,
                   contact_outcome = ?,
                   agent_notes = ?,
                   last_updated_utc = ?
             WHERE synthetic_customer_id = ?
            """,
            (outreach_status, contact_outcome, agent_notes.strip()[:450], now_utc, synthetic_customer_id),
        )
        conn.commit()


def apply_plotly_theme(fig: go.Figure, height: int = 340, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=PALETTE["text_main"], size=14), x=0.01),
        paper_bgcolor=PALETTE["panel"],
        plot_bgcolor=PALETTE["panel"],
        font=dict(color=PALETTE["text_muted"], family="Inter, Segoe UI, sans-serif", size=12),
        height=height,
        margin=dict(l=18, r=18, t=46 if title else 20, b=24),
        legend=dict(
            bgcolor="rgba(11,18,32,0.7)",
            bordercolor=PALETTE["border"],
            borderwidth=1,
            font=dict(color=PALETTE["text_main"], size=11),
        ),
    )
    fig.update_xaxes(gridcolor="rgba(48,65,88,0.45)", zerolinecolor=PALETTE["border"])
    fig.update_yaxes(gridcolor="rgba(48,65,88,0.45)", zerolinecolor=PALETTE["border"])
    return fig


def render_kpi_cards(cards: list[tuple[str, str, str]]) -> None:
    items_html = "".join(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{lbl}</div>'
        f'<div class="kpi-value">{val}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f"</div>"
        for lbl, val, sub in cards
    )
    st.markdown(f'<div class="kpi-grid">{items_html}</div>', unsafe_allow_html=True)


def format_risk_badge_text(risk_band: str) -> str:
    prefix = {
        "Critical Risk": "[CRITICAL]",
        "High Risk": "[HIGH]",
        "Moderate Risk": "[MODERATE]",
        "Low Risk": "[LOW]",
    }
    return f"{prefix.get(risk_band, '[INFO]')} {risk_band}"


SAMPLE_SUBSCRIBER_PROFILES: dict[str, dict[str, Any]] = {
    "High-Risk Onboarding Fiber (2 mo, Month-to-Month, E-Check)": {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": 0,
        "dependents": 0,
        "tenure_months": 2,
        "phone_service": 1,
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract_type": "Month-to-month",
        "paperless_billing": 1,
        "payment_method": "Electronic check",
        "monthly_charges": 91.40,
        "total_charges": 182.80,
    },
    "Unshielded High-Bill Fiber Account (14 mo, No Tech Support)": {
        "gender": "Male",
        "senior_citizen": 1,
        "partner": 1,
        "dependents": 0,
        "tenure_months": 14,
        "phone_service": 1,
        "multiple_lines": "Yes",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract_type": "Month-to-month",
        "paperless_billing": 1,
        "payment_method": "Electronic check",
        "monthly_charges": 99.65,
        "total_charges": 1395.10,
    },
    "Mid-Tenure DSL Developing Subscriber (19 mo, Mailed Check)": {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": 1,
        "dependents": 1,
        "tenure_months": 19,
        "phone_service": 1,
        "multiple_lines": "No",
        "internet_service": "DSL",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "Yes",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract_type": "Month-to-month",
        "paperless_billing": 1,
        "payment_method": "Mailed check",
        "monthly_charges": 51.20,
        "total_charges": 972.80,
    },
    "Loyal Shielded Two-Year Subscriber (58 mo, Auto Credit Card)": {
        "gender": "Male",
        "senior_citizen": 0,
        "partner": 1,
        "dependents": 1,
        "tenure_months": 58,
        "phone_service": 1,
        "multiple_lines": "Yes",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "contract_type": "Two year",
        "paperless_billing": 0,
        "payment_method": "Credit card (automatic)",
        "monthly_charges": 79.50,
        "total_charges": 4611.00,
    },
}


def main() -> None:
    inject_design_system_css()
    ensure_artifacts_ready()

    artifacts = load_static_artifacts()
    cfg = artifacts["config"]
    metrics = artifacts["metrics"]
    fact_df, eval_df, audit_df = load_star_schema_joined()
    bundle = load_cached_model_bundle()

    NAV_OPTIONS = [
        "1. Churn Summary (BI Page 1)",
        "2. Churn Risk Register (BI Page 2)",
        "3. Recovery Action Center (BI Page 3)",
        "4. Live Scoring & What-If Demo",
        "5. Model & Data Quality (BI Page 4)",
        "6. SQL, Power BI Kit & Architecture",
    ]
    if "active_nav" not in st.session_state:
        st.session_state["active_nav"] = NAV_OPTIONS[0]

    def _jump_to(target_view: str) -> None:
        st.session_state["active_nav"] = target_view

    # Sidebar Navigation & Global Slicers
    with st.sidebar:
        st.markdown(f"### {cfg['project']['title']}")
        st.caption(f"Model `{metrics['model_version']}` | Data `{metrics['data_version']}`")

        selected_nav = st.radio(
            "Workspace View",
            options=NAV_OPTIONS,
            key="active_nav",
            help="Navigate across the 4 Power BI analytical views, the interactive scoring simulator, and technical artifacts.",
        )

        st.markdown("---")
        st.markdown("#### Global BI Segment Slicers")
        split_filter = st.multiselect(
            "Dataset Split",
            options=["train", "validation", "test"],
            default=["train", "validation", "test"],
            help="Filter to 'test' to inspect only the 1,409 untouched held-out subscribers.",
        )
        contract_filter = st.multiselect(
            "Contract Commitment",
            options=["Month-to-month", "One year", "Two year"],
            default=["Month-to-month", "One year", "Two year"],
        )
        internet_filter = st.multiselect(
            "Internet Service Tier",
            options=["Fiber optic", "DSL", "No"],
            default=["Fiber optic", "DSL", "No"],
        )
        senior_filter = st.multiselect(
            "Senior Segment",
            options=["Non-Senior", "Senior (65+)"],
            default=["Non-Senior", "Senior (65+)"],
        )

        st.markdown("---")
        author_cfg = cfg.get("author", {})
        if author_cfg.get("github_url"):
            st.markdown(
                f"**Developer Repository Profile**  \n[{author_cfg.get('display_name', 'GitHub')}]({author_cfg['github_url']})"
            )

    # Executive Brief Header with Quick-Jump Demo CTA
    top_col1, top_col2 = st.columns([4.2, 1.2])
    with top_col1:
        st.markdown(
            f"""
            <div class="hero-banner">
                <div class="hero-title">Customer Churn Intelligence &amp; Retention Action Center</div>
                <div class="hero-sub">
                    End-to-end subscription retention workflow from <b>SQL Server staging &amp; validation</b> to
                    <b>calibrated LightGBM risk scoring</b>, <b>TreeSHAP non-causal signals</b>, and
                    <b>Power BI Star Schema &amp; Action Queue operations</b>.
                </div>
                <div class="pill-row">
                    <span class="tech-pill">SQL Server T-SQL + SQLite Mirror</span>
                    <span class="tech-pill">LightGBM (Untouched Test ROC-AUC {metrics['test_metrics_operational']['roc_auc']:.3f})</span>
                    <span class="tech-pill">In-Fold SMOTE vs Calibration Study</span>
                    <span class="tech-pill">TreeSHAP Reason Codes</span>
                    <span class="tech-pill">Power BI Star Schema + DAX</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_col2:
        st.write("")
        st.button(
            "Open Live Scoring Demo",
            type="primary",
            width="stretch",
            on_click=_jump_to,
            args=("4. Live Scoring & What-If Demo",),
        )
        st.button(
            "Open Action Queue",
            width="stretch",
            on_click=_jump_to,
            args=("3. Recovery Action Center (BI Page 3)",),
        )

    # Apply global slicers for BI pages 1 & 2
    filtered_df = fact_df[
        fact_df["dataset_split"].isin(split_filter)
        & fact_df["contract_type"].isin(contract_filter)
        & fact_df["internet_service"].isin(internet_filter)
        & fact_df["senior_segment"].isin(senior_filter)
    ].copy()

    # =========================================================================
    # VIEW 1: EXECUTIVE CHURN SUMMARY (Power BI Page 1)
    # =========================================================================
    if selected_nav == "1. Churn Summary (BI Page 1)":
        st.markdown(
            '<div class="governance-box">'
            "<b>Financial Proxy Governance:</b> Monthly exposure figures measure <i>gross recurring MonthlyCharges (USD/month)</i> "
            "among at-risk subscribers. Without margin accounting or randomized intervention holdouts, this platform reports "
            "<b>exposure proxies</b> and never claims causally saved revenue."
            "</div>",
            unsafe_allow_html=True,
        )

        if filtered_df.empty:
            st.warning("No subscriber records match the current sidebar slicer combination. Reset one or more slicers in the left panel.")
            return

        total_n = len(filtered_df)
        churn_n = int(filtered_df["observed_churn_flag"].sum())
        obs_rate = float(filtered_df["observed_churn_flag"].mean() * 100)
        flagged_n = int(filtered_df["predicted_high_risk_flag"].sum())
        gross_queue_usd = float(filtered_df["monthly_charges_at_risk_proxy"].sum())
        prob_exp_usd = float(filtered_df["expected_monthly_exposure_proxy"].sum())

        render_kpi_cards(
            [
                ("Filtered Subscribers", f"{total_n:,}", f"Of {len(fact_df):,} IBM Benchmark Accounts"),
                ("Observed Churn Rate", f"{obs_rate:.2f}%", f"{churn_n:,} Churned Subscribers"),
                ("Flagged Action Queue", f"{flagged_n:,}", f"{flagged_n/total_n*100:.1f}% (>= {metrics['operational_threshold']:.2f} Cutoff)"),
                ("Gross Bill in Queue (Proxy)", f"${gross_queue_usd:,.0f}/mo", "Flagged MonthlyCharges Sum"),
                ("Prob-Weighted Exposure", f"${prob_exp_usd:,.0f}/mo", "Sum(Calibrated Risk x Monthly Bill)"),
            ]
        )

        c1, c2 = st.columns(2)
        with c1:
            contract_agg = (
                filtered_df.groupby("contract_type", as_index=False)
                .agg(
                    subscribers=("customer_sk", "count"),
                    observed_churn_pct=("observed_churn_flag", lambda s: s.mean() * 100),
                    mean_risk_pct=("churn_probability", lambda s: s.mean() * 100),
                )
            )
            order_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            contract_agg["ord"] = contract_agg["contract_type"].map(order_map)
            contract_agg = contract_agg.sort_values("ord")

            fig_c = go.Figure()
            fig_c.add_trace(
                go.Bar(
                    x=contract_agg["contract_type"],
                    y=contract_agg["observed_churn_pct"],
                    name="Observed Churn (%)",
                    marker_color=PALETTE["secondary"],
                    text=[f"{v:.1f}%" for v in contract_agg["observed_churn_pct"]],
                    textposition="outside",
                )
            )
            fig_c.add_trace(
                go.Bar(
                    x=contract_agg["contract_type"],
                    y=contract_agg["mean_risk_pct"],
                    name="Mean Calibrated Risk (%)",
                    marker_color=PALETTE["primary"],
                    text=[f"{v:.1f}%" for v in contract_agg["mean_risk_pct"]],
                    textposition="outside",
                )
            )
            fig_c.update_layout(barmode="group", yaxis_title="Rate (%)", yaxis_range=[0, max(55.0, float(contract_agg["observed_churn_pct"].max() * 1.25))])
            st.plotly_chart(apply_plotly_theme(fig_c, 350, "Churn Rate by Contract Commitment (%)"), width="stretch")

        with c2:
            cohort_order = ["0-6m Onboarding", "7-12m Early", "13-24m Developing", "25-48m Established", "49m+ Loyal"]
            tenure_agg = (
                filtered_df.groupby("tenure_cohort", as_index=False)
                .agg(
                    subscribers=("customer_sk", "count"),
                    observed_churn_pct=("observed_churn_flag", lambda s: s.mean() * 100),
                    mean_risk_pct=("churn_probability", lambda s: s.mean() * 100),
                )
            )
            tenure_agg["ord"] = tenure_agg["tenure_cohort"].apply(lambda x: cohort_order.index(x) if x in cohort_order else 99)
            tenure_agg = tenure_agg.sort_values("ord")

            fig_t = go.Figure()
            fig_t.add_trace(
                go.Bar(
                    x=tenure_agg["tenure_cohort"],
                    y=tenure_agg["subscribers"],
                    name="Subscribers (Count)",
                    marker_color="rgba(138,180,248,0.28)",
                    yaxis="y",
                )
            )
            fig_t.add_trace(
                go.Scatter(
                    x=tenure_agg["tenure_cohort"],
                    y=tenure_agg["observed_churn_pct"],
                    name="Observed Churn (%)",
                    mode="lines+markers",
                    line=dict(color=PALETTE["critical"], width=2.5),
                    yaxis="y2",
                )
            )
            fig_t.add_trace(
                go.Scatter(
                    x=tenure_agg["tenure_cohort"],
                    y=tenure_agg["mean_risk_pct"],
                    name="Calibrated Risk (%)",
                    mode="lines+markers",
                    line=dict(color=PALETTE["primary"], width=2.5, dash="dot"),
                    yaxis="y2",
                )
            )
            fig_t = apply_plotly_theme(fig_t, 350, "Lifecycle Tenure Cohort Volume vs Attrition Curve")
            fig_t.update_layout(
                yaxis=dict(title="Subscriber Count", showgrid=False),
                yaxis2=dict(title="Attrition / Risk (%)", overlaying="y", side="right", range=[0, 65]),
            )
            st.plotly_chart(fig_t, width="stretch")

        c3, c4 = st.columns(2)
        with c3:
            inet_agg = (
                filtered_df.groupby(["internet_service", "protection_support_count"], as_index=False)
                .agg(churn_pct=("observed_churn_flag", lambda s: s.mean() * 100), n=("customer_sk", "count"))
            )
            fig_i = go.Figure()
            for tier, clr in [("Fiber optic", PALETTE["critical"]), ("DSL", PALETTE["secondary"]), ("No", PALETTE["primary"])]:
                sub = inet_agg[inet_agg["internet_service"] == tier]
                if not sub.empty:
                    fig_i.add_trace(
                        go.Scatter(
                            x=sub["protection_support_count"],
                            y=sub["churn_pct"],
                            mode="lines+markers",
                            name=tier,
                            line=dict(color=clr, width=2.5),
                        )
                    )
            fig_i.update_layout(
                xaxis_title="Active Security/Backup/Protection/TechSupport Add-Ons (0 to 4)",
                yaxis_title="Observed Churn Rate (%)",
            )
            st.plotly_chart(
                apply_plotly_theme(fig_i, 330, "Service Shield Stickiness: Churn vs Active Protection Add-Ons"),
                width="stretch",
            )

        with c4:
            pay_agg = (
                filtered_df.groupby("payment_method", as_index=False)
                .agg(
                    prob_weighted_usd=("expected_monthly_exposure_proxy", "sum"),
                    churn_pct=("observed_churn_flag", lambda s: s.mean() * 100),
                )
                .sort_values("prob_weighted_usd", ascending=True)
            )
            fig_p = go.Figure(
                go.Bar(
                    x=pay_agg["prob_weighted_usd"],
                    y=pay_agg["payment_method"],
                    orientation="h",
                    marker_color=PALETTE["primary"],
                    text=[f"${v:,.0f}/mo ({r:.1f}% churn)" for v, r in zip(pay_agg["prob_weighted_usd"], pay_agg["churn_pct"])],
                    textposition="outside",
                )
            )
            fig_p.update_layout(
                xaxis_title="Probability-Weighted Monthly Exposure Proxy ($ USD/mo)",
                xaxis_range=[0, max(1000.0, float(pay_agg["prob_weighted_usd"].max() * 1.38))],
            )
            st.plotly_chart(apply_plotly_theme(fig_p, 330, "Monthly Exposure Proxy by Payment Channel"), width="stretch")

    # =========================================================================
    # VIEW 2: CHURN RISK REGISTER (Power BI Page 2)
    # =========================================================================
    elif selected_nav == "2. Churn Risk Register (BI Page 2)":
        if filtered_df.empty:
            st.warning("No records match the active sidebar slicers.")
            return

        rb_counts = filtered_df["risk_band"].value_counts()
        render_kpi_cards(
            [
                ("[CRITICAL] >= 0.65", f"{int(rb_counts.get('Critical Risk', 0)):,}", "Within 24h Care Queue"),
                ("[HIGH] 0.45 - 0.64", f"{int(rb_counts.get('High Risk', 0)):,}", "Within 72h Proactive Queue"),
                (f"[MODERATE] {metrics['operational_threshold']:.2f} - 0.44", f"{int(rb_counts.get('Moderate Risk', 0)):,}", "Digital Check-In Pool"),
                ("[LOW] Below Cutoff", f"{int(rb_counts.get('Low Risk', 0)):,}", "No Proactive Call"),
            ]
        )

        col_s1, col_s2 = st.columns([1.05, 1.0])
        with col_s1:
            shap_df = artifacts["shap_global"].head(10).iloc[::-1]
            fig_sh = go.Figure(
                go.Bar(
                    x=shap_df["mean_abs_shap"],
                    y=shap_df["feature_display_name"],
                    orientation="h",
                    marker_color=PALETTE["primary"],
                    text=[f"{v:.3f}" for v in shap_df["mean_abs_shap"]],
                    textposition="outside",
                )
            )
            fig_sh.update_layout(
                xaxis_title="Mean |SHAP| Contribution (Log-Odds Scale — Observational)",
                xaxis_range=[0, float(shap_df["mean_abs_shap"].max() * 1.22)],
            )
            st.plotly_chart(apply_plotly_theme(fig_sh, 350, "Global Associated Risk Signals (TreeSHAP)"), width="stretch")

        with col_s2:
            rb_order = ["Critical Risk", "High Risk", "Moderate Risk", "Low Risk"]
            rb_agg = (
                filtered_df.groupby("risk_band", as_index=False)
                .agg(
                    gross_bill=("monthly_charges_usd", "sum"),
                    prob_weighted=("expected_monthly_exposure_proxy", "sum"),
                    count=("customer_sk", "count"),
                )
            )
            rb_agg["ord"] = rb_agg["risk_band"].apply(lambda x: rb_order.index(x) if x in rb_order else 9)
            rb_agg = rb_agg.sort_values("ord")

            fig_rb = go.Figure()
            fig_rb.add_trace(
                go.Bar(
                    x=rb_agg["risk_band"],
                    y=rb_agg["gross_bill"],
                    name="Total Monthly Bill ($/mo)",
                    marker_color=PALETTE["secondary"],
                )
            )
            fig_rb.add_trace(
                go.Bar(
                    x=rb_agg["risk_band"],
                    y=rb_agg["prob_weighted"],
                    name="Prob-Weighted Exposure ($/mo)",
                    marker_color=PALETTE["primary"],
                )
            )
            fig_rb.update_layout(barmode="group", yaxis_title="USD / Month (Gross Proxy)")
            st.plotly_chart(apply_plotly_theme(fig_rb, 350, "Portfolio Monthly Bill vs Risk-Weighted Exposure by Band"), width="stretch")

        st.markdown("#### Subscriber Churn Risk Register")
        f_col1, f_col2, f_col3 = st.columns([1.3, 1.2, 1.2])
        with f_col1:
            selected_bands = st.multiselect(
                "Filter Risk Bands",
                options=["Critical Risk", "High Risk", "Moderate Risk", "Low Risk"],
                default=["Critical Risk", "High Risk", "Moderate Risk"],
            )
        with f_col2:
            min_bill = st.slider("Minimum Monthly Bill ($ USD/mo)", 0.0, 120.0, 0.0, 5.0)
        with f_col3:
            search_id = st.text_input("Search Synthetic Customer ID (e.g. SYN-CUST-03466)", value="").strip()

        reg_df = filtered_df[
            filtered_df["risk_band"].isin(selected_bands) & (filtered_df["monthly_charges_usd"] >= min_bill)
        ].copy()
        if search_id:
            reg_df = reg_df[reg_df["synthetic_customer_id"].str.contains(search_id.upper(), na=False)]

        reg_df = reg_df.sort_values(["churn_probability", "monthly_charges_usd"], ascending=False)

        if reg_df.empty:
            st.info("No subscribers match the Risk Register filters above.")
        else:
            display_reg = reg_df[
                [
                    "synthetic_customer_id",
                    "risk_band",
                    "churn_probability",
                    "monthly_charges_usd",
                    "expected_monthly_exposure_proxy",
                    "tenure_months",
                    "contract_type",
                    "internet_service",
                    "reason_code_1",
                    "reason_code_2",
                    "dataset_split",
                ]
            ].head(250).copy()
            display_reg["risk_band"] = display_reg["risk_band"].apply(format_risk_badge_text)
            display_reg["churn_probability"] = (display_reg["churn_probability"] * 100).round(1)

            st.dataframe(
                display_reg.rename(
                    columns={
                        "synthetic_customer_id": "Customer ID",
                        "risk_band": "Risk Band",
                        "churn_probability": "Churn Prob (%)",
                        "monthly_charges_usd": "Monthly Bill ($)",
                        "expected_monthly_exposure_proxy": "Expected Exposure ($)",
                        "tenure_months": "Tenure (Mo)",
                        "contract_type": "Contract",
                        "internet_service": "Internet",
                        "reason_code_1": "Signal #1",
                        "reason_code_2": "Signal #2",
                        "dataset_split": "Split",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "Download Filtered Risk Register (CSV)",
                data=dataframe_to_csv_bytes(reg_df),
                file_name="churn_risk_register_filtered.csv",
                mime="text/csv",
            )

    # =========================================================================
    # VIEW 3: RECOVERY ACTION CENTER (Power BI Page 3)
    # =========================================================================
    elif selected_nav == "3. Recovery Action Center (BI Page 3)":
        st.markdown(
            '<div class="governance-box">'
            "<b>Non-Causal Playbook Notice:</b> Recommended retention actions are <i>analytical rule-based playbooks</i> mapped from "
            "observational SHAP risk signals. An elevated risk score does NOT prove that a discount or phone call will prevent churn; "
            "deploy outreach with a randomized holdout control group (uplift testing) before reporting saved revenue."
            "</div>",
            unsafe_allow_html=True,
        )

        queue_df = load_live_action_queue()

        # Interactive Threshold Policy Simulator
        with st.expander("Interactive Capacity & Cost-Threshold Policy Planner (Validation Set Optimization)", expanded=False):
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                cap_pct = st.slider("Max Team Outreach Capacity (% of Base)", 10, 50, 30, 1) / 100.0
            with p_col2:
                call_cost = st.number_input("Assumed Outreach Cost ($ USD / Contact)", min_value=1.0, max_value=100.0, value=8.0, step=1.0)
            with p_col3:
                fn_penalty = st.number_input("Assumed Missed-Churn Scenario Penalty ($ USD)", min_value=10.0, max_value=500.0, value=65.0, step=5.0)

            val_rows = fact_df[fact_df["dataset_split"] == "validation"]
            sim_thresh, sim_curve = optimize_operational_threshold(
                y_val=val_rows["observed_churn_flag"].to_numpy(),
                p_val=val_rows["churn_probability"].to_numpy(),
                capacity_cap_pct=cap_pct,
                outreach_cost_usd=call_cost,
                missed_churn_penalty_usd=fn_penalty,
            )
            test_rows = fact_df[fact_df["dataset_split"] == "test"]
            sim_test_flagged = (test_rows["churn_probability"] >= sim_thresh).mean() * 100
            sim_test_recall = (
                test_rows.loc[test_rows["churn_probability"] >= sim_thresh, "observed_churn_flag"].sum()
                / max(1, test_rows["observed_churn_flag"].sum())
            ) * 100
            st.info(
                f"Policy Simulation Result -> Recommended Cutoff: **{sim_thresh:.2f}** "
                f"(Default warehouse cutoff is `{metrics['operational_threshold']:.2f}`). "
                f"On Untouched Test, `{sim_thresh:.2f}` flags **{sim_test_flagged:.1f}%** of accounts and captures **{sim_test_recall:.1f}%** of actual churners."
            )

        p1_n = int((queue_df["priority_tier"] == "P1 - Immediate (Top 250)").sum())
        pending_n = int((queue_df["outreach_status"] == "Pending Review").sum())
        completed_n = int((queue_df["outreach_status"] != "Pending Review").sum())
        queue_bill_usd = float(queue_df["monthly_charges_usd"].sum())

        render_kpi_cards(
            [
                ("Active Outreach Queue", f"{len(queue_df):,}", f"Threshold >= {metrics['operational_threshold']:.2f}"),
                ("P1 Immediate Desk", f"{p1_n:,}", "Top 250 Priority Score Accounts"),
                ("Pending Agent Review", f"{pending_n:,}", f"{completed_n:,} Logged / Updated"),
                ("Gross Bill in Queue", f"${queue_bill_usd:,.0f}/mo", "Gross Recurring Billing Proxy"),
            ]
        )

        qa1, qa2 = st.columns(2)
        with qa1:
            pb_agg = queue_df["recommended_playbook"].value_counts().reset_index()
            pb_agg.columns = ["playbook", "count"]
            pb_agg = pb_agg.iloc[::-1]
            fig_pb = go.Figure(
                go.Bar(
                    x=pb_agg["count"],
                    y=pb_agg["playbook"],
                    orientation="h",
                    marker_color=PALETTE["primary"],
                    text=pb_agg["count"],
                    textposition="outside",
                )
            )
            fig_pb.update_layout(xaxis_range=[0, float(pb_agg["count"].max() * 1.22)], xaxis_title="Queued Subscribers")
            st.plotly_chart(apply_plotly_theme(fig_pb, 320, "Action Queue by Analytical Playbook Suggestion"), width="stretch")

        with qa2:
            tm_agg = queue_df["assigned_team"].value_counts().reset_index()
            tm_agg.columns = ["team", "count"]
            tm_agg = tm_agg.iloc[::-1]
            fig_tm = go.Figure(
                go.Bar(
                    x=tm_agg["count"],
                    y=tm_agg["team"],
                    orientation="h",
                    marker_color=PALETTE["secondary"],
                    text=tm_agg["count"],
                    textposition="outside",
                )
            )
            fig_tm.update_layout(xaxis_range=[0, float(tm_agg["count"].max() * 1.22)], xaxis_title="Assigned Accounts")
            st.plotly_chart(apply_plotly_theme(fig_tm, 320, "Workload Allocation Across Specialist Care Desks"), width="stretch")

        # Queue Filter + Agent Outcome Logging Form
        q_f1, q_f2, q_f3 = st.columns(3)
        with q_f1:
            tier_sel = st.multiselect(
                "Priority Tier",
                options=sorted(queue_df["priority_tier"].unique()),
                default=sorted(queue_df["priority_tier"].unique()),
            )
        with q_f2:
            desk_sel = st.multiselect(
                "Assigned Care Desk",
                options=sorted(queue_df["assigned_team"].unique()),
                default=sorted(queue_df["assigned_team"].unique()),
            )
        with q_f3:
            status_sel = st.multiselect(
                "Outreach Status",
                options=sorted(queue_df["outreach_status"].unique()),
                default=sorted(queue_df["outreach_status"].unique()),
            )

        q_view = queue_df[
            queue_df["priority_tier"].isin(tier_sel)
            & queue_df["assigned_team"].isin(desk_sel)
            & queue_df["outreach_status"].isin(status_sel)
        ].copy()

        st.dataframe(
            q_view[
                [
                    "priority_rank",
                    "synthetic_customer_id",
                    "priority_tier",
                    "risk_band",
                    "churn_probability",
                    "monthly_charges_usd",
                    "primary_reason_code",
                    "secondary_reason_code",
                    "recommended_playbook",
                    "assigned_team",
                    "outreach_status",
                    "contact_outcome",
                    "agent_notes",
                ]
            ].head(200),
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Log Care Specialist Outreach Outcome")
        log_c1, log_c2, log_c3, log_c4 = st.columns([1.2, 1.1, 1.3, 1.6])
        candidate_ids = q_view["synthetic_customer_id"].head(300).tolist() or queue_df["synthetic_customer_id"].head(50).tolist()
        with log_c1:
            target_cid = st.selectbox("Select Queued Customer ID", options=candidate_ids)
        with log_c2:
            new_status = st.selectbox(
                "Follow-Up Status",
                options=["Pending Review", "Contacted", "Follow-Up Scheduled", "Resolved", "Holdout Control (No Call)"],
            )
        with log_c3:
            new_outcome = st.selectbox(
                "Contact Outcome",
                options=[
                    "Not Contacted Yet",
                    "Accepted Plan / Bundle Review",
                    "Technical Check Scheduled",
                    "Declined Offer — Remains MTM",
                    "Voicemail / No Answer",
                    "Assigned to A/B Holdout Control",
                ],
            )
        with log_c4:
            note_txt = st.text_input("Care Agent Note (Non-PII only)", placeholder="e.g. Scheduled Wi-Fi mesh check; do not enter personal PII")

        if st.button("Save Outreach Log to Warehouse", type="primary"):
            update_action_queue_record(target_cid, new_status, new_outcome, note_txt)
            st.success(f"Updated outreach workflow record for `{target_cid}` ({new_status} — {new_outcome}).")
            st.rerun()

        st.markdown("#### Operational Exports")
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            st.download_button(
                "Download Action Queue (CSV)",
                data=dataframe_to_csv_bytes(q_view),
                file_name="retention_action_queue.csv",
                mime="text/csv",
                width="stretch",
            )
        with ex2:
            xlsx_bytes = action_pack_to_excel_bytes(q_view, eval_df, audit_df)
            if xlsx_bytes is not None:
                st.download_button(
                    "Download Full BI Workbook (.xlsx)",
                    data=xlsx_bytes,
                    file_name="churn_retention_operations_pack.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )
        with ex3:
            pdf_bytes = generate_executive_pdf_brief(metrics, q_view)
            if pdf_bytes is not None:
                st.download_button(
                    "Download Executive Summary (.pdf)",
                    data=pdf_bytes,
                    file_name="Churn_Intelligence_Executive_Brief.pdf",
                    mime="application/pdf",
                    width="stretch",
                )

    # =========================================================================
    # VIEW 4: LIVE CUSTOMER SCORING & WHAT-IF DEMO
    # =========================================================================
    elif selected_nav == "4. Live Scoring & What-If Demo":
        st.markdown("#### Interactive Subscriber Risk & Explainability Simulator")
        selected_profile_name = st.selectbox(
            "Load Sample Subscriber Profile (Ready-to-Test Presets)",
            options=list(SAMPLE_SUBSCRIBER_PROFILES.keys()),
        )
        preset = SAMPLE_SUBSCRIBER_PROFILES[selected_profile_name]

        with st.form("live_scoring_form"):
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                contract_type = st.selectbox(
                    "Contract Term",
                    ["Month-to-month", "One year", "Two year"],
                    index=["Month-to-month", "One year", "Two year"].index(preset["contract_type"]),
                )
                tenure_months = st.number_input(
                    "Tenure (Months, 0 - 120)",
                    min_value=0,
                    max_value=120,
                    value=int(preset["tenure_months"]),
                    step=1,
                )
                monthly_charges = st.number_input(
                    "Current Monthly Bill ($ USD)",
                    min_value=0.0,
                    max_value=500.0,
                    value=float(preset["monthly_charges"]),
                    step=1.0,
                )
                default_tc = 0.0 if tenure_months == 0 else float(preset["total_charges"])
                total_charges = st.number_input(
                    "Cumulative Total Charges ($ USD)",
                    min_value=0.0,
                    max_value=50000.0,
                    value=default_tc,
                    step=10.0,
                    help="Automatically set to $0.00 when Tenure = 0 months.",
                )

            with f2:
                internet_service = st.selectbox(
                    "Internet Access Tier",
                    ["Fiber optic", "DSL", "No"],
                    index=["Fiber optic", "DSL", "No"].index(preset["internet_service"]),
                )
                payment_method = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    index=[
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ].index(preset["payment_method"]),
                )
                paperless_billing = st.selectbox(
                    "Paperless Billing",
                    ["Yes", "No"],
                    index=0 if preset["paperless_billing"] == 1 else 1,
                )
                phone_service = st.selectbox(
                    "Voice Phone Service",
                    ["Yes", "No"],
                    index=0 if preset["phone_service"] == 1 else 1,
                )

            with f3:
                online_security = st.selectbox("Online Security Add-On", ["No", "Yes"], index=1 if preset["online_security"] == "Yes" else 0)
                online_backup = st.selectbox("Online Backup Add-On", ["No", "Yes"], index=1 if preset["online_backup"] == "Yes" else 0)
                device_protection = st.selectbox("Device Protection Add-On", ["No", "Yes"], index=1 if preset["device_protection"] == "Yes" else 0)
                tech_support = st.selectbox("Tech Support Add-On", ["No", "Yes"], index=1 if preset["tech_support"] == "Yes" else 0)

            with f4:
                streaming_tv = st.selectbox("Streaming TV Add-On", ["No", "Yes"], index=1 if preset["streaming_tv"] == "Yes" else 0)
                streaming_movies = st.selectbox("Streaming Movies Add-On", ["No", "Yes"], index=1 if preset["streaming_movies"] == "Yes" else 0)
                multiple_lines = st.selectbox("Multiple Phone Lines", ["No", "Yes"], index=1 if preset["multiple_lines"] == "Yes" else 0)
                senior_citizen = st.selectbox("Senior Citizen (65+)", ["No", "Yes"], index=1 if preset["senior_citizen"] == 1 else 0)

            submitted = st.form_submit_button("Evaluate Calibrated Churn Risk & SHAP Signals", type="primary")

        input_payload = {
            "gender": preset["gender"],
            "senior_citizen": 1 if senior_citizen == "Yes" else 0,
            "partner": preset["partner"],
            "dependents": preset["dependents"],
            "tenure_months": tenure_months,
            "phone_service": 1 if phone_service == "Yes" else 0,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract_type": contract_type,
            "paperless_billing": 1 if paperless_billing == "Yes" else 0,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": 0.0 if tenure_months == 0 else total_charges,
        }

        try:
            result = score_single_customer_live(input_payload, bundle=bundle)
        except ValueError as exc:
            st.error(f"Input Validation Error: {exc}")
            return

        if submitted or True:
            render_kpi_cards(
                [
                    ("Calibrated Churn Risk", f"{result['churn_probability']*100:.1f}%", f"Operational Cutoff: {result['operational_threshold']*100:.0f}%"),
                    ("Assigned Risk Band", format_risk_badge_text(result["risk_band"]), "Flagged for Queue" if result["flagged_for_outreach"] else "Below Outreach Threshold"),
                    ("Monthly Bill Exposure", f"${result['monthly_charges_usd']:.2f}/mo", f"Prob-Weighted: ${result['expected_monthly_exposure_proxy_usd']:.2f}/mo"),
                    ("Suggested Care Desk", result["assigned_team"], result["recommended_playbook"][:32]),
                ]
            )

            r_col1, r_col2 = st.columns([1.2, 1.0])
            with r_col1:
                drv_df = pd.DataFrame(result["shap_drivers"]).head(8).iloc[::-1]
                bar_colors = [PALETTE["critical"] if v > 0 else PALETTE["primary"] for v in drv_df["shap_contribution"]]
                fig_local = go.Figure(
                    go.Bar(
                        x=drv_df["shap_contribution"],
                        y=drv_df["concept"],
                        orientation="h",
                        marker_color=bar_colors,
                        text=[f"{v:+.3f}" for v in drv_df["shap_contribution"]],
                        textposition="outside",
                    )
                )
                fig_local.update_layout(
                    xaxis_title="Customer-Specific SHAP Contribution (Log-Odds; + Pushes Risk Up, - Offsets Risk)",
                )
                st.plotly_chart(
                    apply_plotly_theme(fig_local, 340, "Subscriber-Specific TreeSHAP Signal Breakdown (Observational)"),
                    width="stretch",
                )

            with r_col2:
                st.markdown("#### Analytical Playbook & Signal Summary")
                st.markdown(
                    f"""
                    <div class="governance-box">
                        <b>Suggested Playbook (Hypothesis):</b> {result['recommended_playbook']}<br/>
                        <b>Assigned Desk:</b> {result['assigned_team']}<br/>
                        <b>Analytical Rationale:</b> {result['playbook_analytical_note']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                top_pos = [d for d in result["shap_drivers"] if d["shap_contribution"] > 0][:3]
                if top_pos:
                    st.markdown("**Top Associated Risk Indicators (Non-Causal):**")
                    for d in top_pos:
                        st.markdown(f"- `{d['reason_code']}` — {d['description_non_causal']} (`SHAP {d['shap_contribution']:+.3f}`)")
                else:
                    st.markdown("All major subscription attributes currently offset churn risk relative to the portfolio baseline.")

                # Observational Model Profile Comparison (What-If Sensitivity)
                if input_payload["contract_type"] == "Month-to-month":
                    alt_payload = dict(input_payload, contract_type="One year")
                    alt_res = score_single_customer_live(alt_payload, bundle=bundle)
                    st.caption(
                        f"Model Sensitivity Note (Observational, Non-Causal): A comparable subscriber profile on a `One year` contract "
                        f"scores at **{alt_res['churn_probability']*100:.1f}%** predicted risk (vs **{result['churn_probability']*100:.1f}%** current). "
                        f"Real-world conversion & retention lift must be verified via controlled A/B testing."
                    )

    # =========================================================================
    # VIEW 5: MODEL & DATA QUALITY GOVERNANCE (Power BI Page 4)
    # =========================================================================
    elif selected_nav == "5. Model & Data Quality (BI Page 4)":
        tm_op = metrics["test_metrics_operational"]
        tm_05 = metrics["test_metrics_default_050"]
        ci = metrics["test_bootstrap_95ci"]

        render_kpi_cards(
            [
                ("Champion Model", metrics["champion_model_name"], "Regularized Natural Prior"),
                ("Untouched Test ROC-AUC", f"{tm_op['roc_auc']:.4f}", f"95% CI [{ci['roc_auc_95ci'][0]:.4f}, {ci['roc_auc_95ci'][1]:.4f}]"),
                ("Untouched Test PR-AUC", f"{tm_op['pr_auc']:.4f}", f"95% CI [{ci['pr_auc_95ci'][0]:.4f}, {ci['pr_auc_95ci'][1]:.4f}]"),
                ("Brier Calibration", f"{tm_op['brier_score']:.4f}", f"Test ECE = {tm_op['ece']:.4f} (Mean P={tm_op['mean_predicted_prob']:.3f})"),
                ("Operational Recall (@0.39)", f"{tm_op['recall']*100:.1f}%", f"vs {tm_05['recall']*100:.1f}% at Default 0.50"),
            ]
        )

        m_c1, m_c2 = st.columns(2)
        with m_c1:
            comp_df = artifacts["comparison"]
            fig_m = go.Figure()
            labels = [f"{r['model_name']} ({r['imbalance_strategy'].split()[0]})" for _, r in comp_df.iterrows()]
            fig_m.add_trace(
                go.Bar(x=comp_df["val_pr_auc"], y=labels, orientation="h", name="Validation PR-AUC (Higher)", marker_color=PALETTE["primary"])
            )
            fig_m.add_trace(
                go.Bar(x=comp_df["val_ece"], y=labels, orientation="h", name="Calibration Error ECE (Lower)", marker_color=PALETTE["critical"])
            )
            fig_m.update_layout(barmode="group", xaxis_title="Score / Calibration Error")
            st.plotly_chart(
                apply_plotly_theme(fig_m, 370, "Ablation Study: Natural Prior vs Class-Weight vs In-Fold SMOTE"),
                width="stretch",
            )

        with m_c2:
            cal_df = artifacts["calibration_curve"]
            fig_cal = go.Figure()
            fig_cal.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal Calibration", line=dict(color=PALETTE["text_muted"], dash="dash"))
            )
            fig_cal.add_trace(
                go.Scatter(
                    x=cal_df["bin_mean_predicted_probability"],
                    y=cal_df["bin_observed_churn_frequency"],
                    mode="lines+markers",
                    name="Champion LightGBM (Untouched Test)",
                    line=dict(color=PALETTE["primary"], width=3),
                )
            )
            fig_cal.update_layout(
                xaxis_title="Mean Predicted Churn Probability",
                yaxis_title="Observed Churn Rate in Bin",
                xaxis_range=[0, 1],
                yaxis_range=[0, 1.05],
            )
            st.plotly_chart(apply_plotly_theme(fig_cal, 370, "Probability Calibration Curve (Untouched Test, n=1,409)"), width="stretch")

        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            st.markdown(f"#### Threshold Comparison on Untouched Test (`n = {tm_op['tp']+tm_op['fp']+tm_op['tn']+tm_op['fn']:,}`)")
            th_comp_df = pd.DataFrame(
                [
                    {
                        "Policy": "Default Cutoff (0.50)",
                        "Queue Flagged %": f"{tm_05['flagged_rate']*100:.1f}%",
                        "Precision": f"{tm_05['precision']*100:.1f}%",
                        "Recall (Churners Caught)": f"{tm_05['recall']*100:.1f}%",
                        "Natural Test F1": f"{tm_05['f1']:.4f}",
                        "TP / FP / FN / TN": f"{tm_05['tp']} / {tm_05['fp']} / {tm_05['fn']} / {tm_05['tn']}",
                    },
                    {
                        "Policy": f"Operational Cutoff ({tm_op['threshold']:.2f})",
                        "Queue Flagged %": f"{tm_op['flagged_rate']*100:.1f}%",
                        "Precision": f"{tm_op['precision']*100:.1f}%",
                        "Recall (Churners Caught)": f"{tm_op['recall']*100:.1f}%",
                        "Natural Test F1": f"{tm_op['f1']:.4f}",
                        "TP / FP / FN / TN": f"{tm_op['tp']} / {tm_op['fp']} / {tm_op['fn']} / {tm_op['tn']}",
                    },
                ]
            )
            st.dataframe(th_comp_df, width="stretch", hide_index=True)

        with cm_col2:
            st.markdown("#### SQL Staging & Clean Layer Quality Audit")
            st.dataframe(
                audit_df[["check_stage", "check_name", "metric_value", "check_status", "notes"]],
                width="stretch",
                hide_index=True,
            )

        st.markdown("#### Full Candidate Evaluation Registry (5-Fold Train CV + Validation)")
        st.dataframe(artifacts["comparison"], width="stretch", hide_index=True)

    # =========================================================================
    # VIEW 6: ARCHITECTURE, SQL & POWER BI KIT
    # =========================================================================
    elif selected_nav == "6. SQL, Power BI Kit & Architecture":
        st.markdown("#### Power BI Semantic Star Schema & T-SQL Warehouse Artifacts")
        st.markdown(
            '<div class="governance-box">'
            "<b>Native Power BI (.pbix) Transparency Note:</b> Binary <code>.pbix</code> files require the proprietary Windows "
            "Power BI Desktop VertiPaq engine. Rather than disguising a non-native file with a fake <code>.pbix</code> extension, "
            "this repository ships the complete <b>8-table Star Schema Mart</b>, <b>5 SQL Server T-SQL scripts</b>, "
            "<b>25 DAX measures</b>, <b>Power Query (M) connectors</b>, <b>Dark Theme JSON</b>, and <b>4-Page Executed Reference Snapshots</b>."
            "</div>",
            unsafe_allow_html=True,
        )

        svg_path = POWERBI_DIR / "MODEL_SCHEMA_DIAGRAM.svg"
        if svg_path.exists():
            st.image(svg_path.read_text(encoding="utf-8"), width="stretch")

        dl1, dl2, dl3 = st.columns(3)
        dax_path = POWERBI_DIR / "dax" / "measures_library.dax"
        theme_path = POWERBI_DIR / "theme" / "ChurnIntelligenceDarkTheme.json"
        snap_pdf_path = SNAPSHOTS_DIR / "PowerBI_Report_Pages_Preview.pdf"

        with dl1:
            if dax_path.exists():
                st.download_button(
                    "Download DAX Measures Library (.dax)",
                    data=dax_path.read_bytes(),
                    file_name="measures_library.dax",
                    mime="text/plain",
                    width="stretch",
                )
        with dl2:
            if theme_path.exists():
                st.download_button(
                    "Download Power BI Dark Theme (.json)",
                    data=theme_path.read_bytes(),
                    file_name="ChurnIntelligenceDarkTheme.json",
                    mime="application/json",
                    width="stretch",
                )
        with dl3:
            if snap_pdf_path.exists():
                st.download_button(
                    "Download 4-Page Power BI Layout Preview (.pdf)",
                    data=snap_pdf_path.read_bytes(),
                    file_name="PowerBI_Report_Pages_Preview.pdf",
                    mime="application/pdf",
                    width="stretch",
                )

        st.markdown("#### Executed 4-Page Power BI Reference Layout Snapshots")
        s_cols = st.columns(2)
        snap_files = [
            ("Page 1 — Executive Churn Summary", SNAPSHOTS_DIR / "page1_churn_summary.png"),
            ("Page 2 — Churn Risk Register", SNAPSHOTS_DIR / "page2_churn_risk_register.png"),
            ("Page 3 — Recovery Action Center", SNAPSHOTS_DIR / "page3_recovery_action_center.png"),
            ("Page 4 — Model & Data Quality", SNAPSHOTS_DIR / "page4_model_and_data_quality.png"),
        ]
        for idx, (caption_txt, spath) in enumerate(snap_files):
            if spath.exists():
                with s_cols[idx % 2]:
                    st.image(str(spath), caption=caption_txt, width="stretch")

        st.markdown("#### Inspect Versioned SQL Server (T-SQL) Scripts & DAX")
        sql_files = sorted(SQL_DIR.glob("*.sql"))
        selected_sql = st.selectbox("Select SQL Server Script", [p.name for p in sql_files])
        if selected_sql:
            st.code((SQL_DIR / selected_sql).read_text(encoding="utf-8"), language="sql")

        st.markdown("#### Key Limitations & Scientific Guardrails")
        st.markdown(
            """
            - **Public Educational Benchmark:** The IBM Telco Customer Churn dataset (`7,043` rows, `21` columns) is a static cross-sectional snapshot without timestamps, ticket logs, usage telemetry, or competitor pricing. Results illustrate methodology and BI workflow, not universal telecom performance.
            - **Observational Risk vs Causal Uplift:** High predicted churn risk does not guarantee a customer can be persuaded to stay, nor that offering a discount is profitable. Production retention programs require randomized holdout uplift testing.
            - **Untouched Natural Test vs SMOTE F1:** Some public repositories report `F1 ≈ 0.90` after applying SMOTE prior to train/test splitting or evaluating on oversampled test sets. Here, SMOTE is strictly isolated inside training CV folds, and all final metrics (`ROC-AUC = 0.8478`, `PR-AUC = 0.6640`, `F1 = 0.6337` at `0.39` threshold) are reported on the **untouched natural test split (`26.54%` churn)**.
            - **Gross Billing Exposure Proxy:** `MonthlyCharges` measures gross monthly recurring billing before service costs, taxes, and margin.
            """
        )


if __name__ == "__main__":
    main()
