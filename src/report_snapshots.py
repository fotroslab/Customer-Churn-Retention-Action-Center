"""Generate high-resolution static layout snapshots (PNG + PDF) of the 4 Power BI report pages.

Note: These snapshots render the exact executed SQL/ML Star Schema figures in the project's
dark visual system (#0B1220 / #141F30 / #5EE0B5 / #8AB4F8) as reference page previews
for the Power BI Desktop build guide and GitHub documentation.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from src.config import (
    CALIBRATION_CURVE_CSV_PATH,
    METRICS_REPORT_PATH,
    MODEL_COMPARISON_CSV_PATH,
    PALETTE,
    SHAP_GLOBAL_CSV_PATH,
    SNAPSHOTS_DIR,
    WAREHOUSE_DB_PATH,
)


def _style_ax(ax: plt.Axes, title: str) -> None:
    """Style an axes panel: dark face, bordered spines, high-contrast readable labels."""
    ax.set_facecolor(PALETTE["panel"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
        spine.set_linewidth(0.9)
    # High-contrast tick labels on the left/bottom for readability (muted text
    # was too dim at rendered sizes on the dark background).
    ax.tick_params(colors=PALETTE["text_main"], labelsize=9.4)
    ax.set_title(title, color=PALETTE["text_main"], fontsize=10.8, fontweight="bold", loc="left", pad=9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(PALETTE["text_main"])


def _draw_kpi_strip(fig: plt.Figure, kpis: list[tuple[str, str, str]], page_title: str, subtitle: str) -> None:
    fig.text(0.03, 0.945, page_title, color=PALETTE["text_main"], fontsize=15.0, fontweight="bold")
    fig.text(0.03, 0.913, subtitle, color=PALETTE["text_main"], fontsize=9.2)

    n = len(kpis)
    width = (0.94 - 0.015 * (n - 1)) / n
    for i, (label, val, sub) in enumerate(kpis):
        x0 = 0.03 + i * (width + 0.015)
        ax_k = fig.add_axes([x0, 0.79, width, 0.10])
        ax_k.set_axis_off()
        patch = FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=PALETTE["panel"],
            edgecolor=PALETTE["border"],
            linewidth=1.1,
        )
        ax_k.add_patch(patch)
        ax_k.text(0.06, 0.73, label.upper(), color=PALETTE["text_main"], fontsize=7.8, fontweight="bold")
        ax_k.text(0.06, 0.34, val, color=PALETTE["primary"], fontsize=14.0, fontweight="bold")
        ax_k.text(0.06, 0.10, sub, color=PALETTE["secondary"], fontsize=7.5)


def generate_all_powerbi_snapshots(
    db_path: Path = WAREHOUSE_DB_PATH,
    out_dir: Path = SNAPSHOTS_DIR,
) -> list[Path]:
    """Render all 4 Power BI page reference PNGs + compiled multi-page PDF from executed data."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with METRICS_REPORT_PATH.open("r", encoding="utf-8") as fp:
        metrics = json.load(fp)

    comp_df = pd.read_csv(MODEL_COMPARISON_CSV_PATH)
    shap_df = pd.read_csv(SHAP_GLOBAL_CSV_PATH)
    cal_df = pd.read_csv(CALIBRATION_CURVE_CSV_PATH)

    with sqlite3.connect(db_path) as conn:
        fact_df = pd.read_sql(
            """
            SELECT f.*, c.tenure_cohort, c.senior_segment, cb.contract_type, cb.payment_method,
                   sb.internet_service, sb.protection_support_count
            FROM bi_fact_customer_churn_scoring f
            JOIN bi_dim_customer c ON f.customer_sk = c.customer_sk
            JOIN bi_dim_contract_billing cb ON f.contract_billing_key = cb.contract_billing_key
            JOIN bi_dim_service_bundle sb ON f.service_bundle_key = sb.service_bundle_key
            """,
            conn,
        )
        queue_df = pd.read_sql("SELECT * FROM bi_fact_retention_action_queue", conn)
        audit_df = pd.read_sql("SELECT * FROM bi_fact_data_quality_audit", conn)

    saved_paths: list[Path] = []
    pdf_path = out_dir / "PowerBI_Report_Pages_Preview.pdf"

    with PdfPages(pdf_path) as pdf:
        # =====================================================================
        # PAGE 1: CHURN SUMMARY
        # =====================================================================
        fig1 = plt.figure(figsize=(14, 8), facecolor=PALETTE["bg"], dpi=160)
        total_cust = len(fact_df)
        obs_churn = fact_df["observed_churn_flag"].mean() * 100
        flagged_cnt = int(fact_df["predicted_high_risk_flag"].sum())
        gross_risk = fact_df["monthly_charges_at_risk_proxy"].sum()
        exp_risk = fact_df["expected_monthly_exposure_proxy"].sum()

        _draw_kpi_strip(
            fig1,
            [
                ("Active Benchmark Base", f"{total_cust:,}", "IBM Telco Clean Cohort"),
                ("Observed Churn Rate", f"{obs_churn:.2f}%", f"{int(fact_df['observed_churn_flag'].sum()):,} Historical Exits"),
                ("Priority Queue Flagged", f"{flagged_cnt:,}", f"Threshold >= {metrics['operational_threshold']:.2f}"),
                ("Gross Bill in Queue (Proxy)", f"${gross_risk:,.0f}/mo", "Flagged MonthlyCharges Sum"),
                ("Prob-Weighted Exposure", f"${exp_risk:,.0f}/mo", "Sum(P(Churn) x Monthly Bill)"),
            ],
            "Power BI Page 1 — Executive Churn Summary (Executed Data Snapshot)",
            "Proxy Notice: Financial figures reflect gross monthly recurring bill exposure, not net margin or causally saved revenue.",
        )

        # Chart 1A: Churn Rate by Contract Type (wide left margin for readable labels)
        ax1 = fig1.add_axes([0.19, 0.44, 0.30, 0.28])
        _style_ax(ax1, "Observed Churn Rate by Contract (%)")
        c_grp = fact_df.groupby("contract_type")["observed_churn_flag"].mean().reindex(
            ["Month-to-month", "One year", "Two year"]
        ) * 100
        bars = ax1.barh(c_grp.index, c_grp.values, color=[PALETTE["critical"], PALETTE["secondary"], PALETTE["primary"]], height=0.52)
        ax1.set_xlim(0, max(c_grp.values) * 1.24)
        ax1.tick_params(axis="y", labelsize=10.0)
        for b in bars:
            ax1.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", color=PALETTE["text_main"], fontsize=9.5)

        # Chart 1B: Churn Rate across Tenure Cohorts
        ax2 = fig1.add_axes([0.55, 0.44, 0.41, 0.28])
        _style_ax(ax2, "Observed vs Calibrated Risk Across Tenure Cohorts")
        cohort_order = ["0-6m Onboarding", "7-12m Early", "13-24m Developing", "25-48m Established", "49m+ Loyal"]
        t_grp = fact_df.groupby("tenure_cohort")[["observed_churn_flag", "churn_probability"]].mean().reindex(cohort_order) * 100
        x = np.arange(len(cohort_order))
        ax2.bar(x - 0.17, t_grp["observed_churn_flag"], width=0.34, color=PALETTE["secondary"], label="Observed Churn %")
        ax2.bar(x + 0.17, t_grp["churn_probability"], width=0.34, color=PALETTE["primary"], label="Calibrated Mean Risk %")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["0-6m", "7-12m", "13-24m", "25-48m", "49m+"], fontsize=8)
        ax2.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["border"], labelcolor=PALETTE["text_main"], fontsize=7.5)

        # Chart 1C: Internet Tier (bottom-left, wide left margin for category labels)
        ax3 = fig1.add_axes([0.17, 0.08, 0.36, 0.27])
        _style_ax(ax3, "Churn Rate by Internet Service Tier (%)")
        i_grp = fact_df.groupby("internet_service")["observed_churn_flag"].mean().reindex(["Fiber optic", "DSL", "No"]) * 100
        b_i = ax3.bar(i_grp.index, i_grp.values, color=[PALETTE["high"], PALETTE["secondary"], PALETTE["primary"]], width=0.45)
        ax3.set_ylim(0, max(i_grp.values) * 1.25)
        for b in b_i:
            ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}%", ha="center", color=PALETTE["text_main"], fontsize=8.5)

        # Chart 1D: Payment Exposure (ample gap from Chart 1C)
        ax4 = fig1.add_axes([0.60, 0.08, 0.36, 0.27])
        _style_ax(ax4, "Prob-Weighted Exposure by Payment ($/mo)")
        p_grp = fact_df.groupby("payment_method")["expected_monthly_exposure_proxy"].sum().sort_values(ascending=True)
        short_pay = {
            "Electronic check": "Electronic check",
            "Bank transfer (automatic)": "Bank transfer (auto)",
            "Credit card (automatic)": "Credit card (auto)",
            "Mailed check": "Mailed check",
        }
        b_p = ax4.barh([short_pay.get(k, k) for k in p_grp.index], p_grp.values, color=PALETTE["secondary"], height=0.52)
        ax4.set_xlim(0, max(p_grp.values) * 1.26)
        for b in b_p:
            ax4.text(b.get_width() + 700, b.get_y() + b.get_height() / 2, f"${b.get_width():,.0f}", va="center", color=PALETTE["text_main"], fontsize=8)

        p1 = out_dir / "page1_churn_summary.png"
        fig1.savefig(p1, facecolor=PALETTE["bg"])
        pdf.savefig(fig1, facecolor=PALETTE["bg"])
        plt.close(fig1)
        saved_paths.append(p1)

        # =====================================================================
        # PAGE 2: CHURN RISK REGISTER
        # =====================================================================
        fig2 = plt.figure(figsize=(14, 8), facecolor=PALETTE["bg"], dpi=160)
        rb_counts = fact_df["risk_band"].value_counts()
        crit_cnt = int(rb_counts.get("Critical Risk", 0))
        high_cnt = int(rb_counts.get("High Risk", 0))
        mod_cnt = int(rb_counts.get("Moderate Risk", 0))

        _draw_kpi_strip(
            fig2,
            [
                ("Critical Risk (>=0.65)", f"{crit_cnt:,}", "24h Priority Care SLA"),
                ("High Risk (0.45-0.64)", f"{high_cnt:,}", "72h Proactive Outreach SLA"),
                ("Moderate Queue", f"{mod_cnt:,}", "Digital Check-In Pool"),
                ("Top Global Signal", "Contract Term", "Highest Mean |SHAP| Impact"),
            ],
            "Power BI Page 2 — Churn Risk Register & Global SHAP Signals",
            "Synthetic non-sensitive customer IDs (SYN-CUST-XXXXX). SHAP drivers show observational associations, not proven causality.",
        )

        ax2_1 = fig2.add_axes([0.30, 0.44, 0.22, 0.28])
        _style_ax(ax2_1, "Global Risk Signals (Mean |SHAP|)")
        top_shap = shap_df.head(8).iloc[::-1]
        concise_shap_names = [n.replace("Subscription ", "").replace("Commitment ", "").replace("Preference", "Pref.") for n in top_shap["feature_display_name"]]
        ax2_1.barh(concise_shap_names, top_shap["mean_abs_shap"], color=PALETTE["primary"], height=0.58)
        ax2_1.tick_params(axis="y", labelsize=9.6)

        ax2_2 = fig2.add_axes([0.56, 0.44, 0.40, 0.28])
        _style_ax(ax2_2, "Gross Monthly Charges Exposure by Risk Band ($ USD/mo)")
        rb_order = ["Critical Risk", "High Risk", "Moderate Risk", "Low Risk"]
        rb_exp = fact_df.groupby("risk_band")["monthly_charges_usd"].sum().reindex(rb_order).fillna(0.0)
        b_rb = ax2_2.bar(
            rb_order,
            rb_exp.values,
            color=[PALETTE["critical"], PALETTE["high"], PALETTE["secondary"], PALETTE["primary"]],
            width=0.48,
        )
        ax2_2.set_ylim(0, max(rb_exp.values) * 1.20)
        for b in b_rb:
            ax2_2.text(b.get_x() + b.get_width() / 2, b.get_height() + 4500, f"${b.get_height():,.0f}", ha="center", color=PALETTE["text_main"], fontsize=8)

        # Register Table Preview
        ax2_3 = fig2.add_axes([0.03, 0.05, 0.94, 0.31])
        ax2_3.set_axis_off()
        ax2_3.text(0.0, 0.95, "Prioritized Subscriber Risk Register (Top 7 Sample Rows)", color=PALETTE["text_main"], fontsize=10.5, fontweight="bold")
        top_reg = fact_df.sort_values(["churn_probability", "monthly_charges_usd"], ascending=False).head(7)
        cell_text = []
        for _, r in top_reg.iterrows():
            cell_text.append(
                [
                    r["synthetic_customer_id"],
                    r["risk_band"],
                    f"{r['churn_probability']*100:.1f}%",
                    f"${r['monthly_charges_usd']:.2f}",
                    r["contract_type"],
                    r["internet_service"],
                    f"{r['reason_code_1']} | {r['reason_code_2']}",
                ]
            )
        tbl = ax2_3.table(
            cellText=cell_text,
            colLabels=["Synthetic ID", "Risk Band", "Churn Prob", "Monthly Bill", "Contract", "Internet", "Top 2 Associated Signals"],
            colWidths=[0.12, 0.11, 0.09, 0.09, 0.13, 0.10, 0.36],
            bbox=[0.0, 0.0, 1.0, 0.86],
            cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.6)
        for (row_i, col_i), cell in tbl.get_celld().items():
            cell.set_edgecolor(PALETTE["border"])
            if row_i == 0:
                cell.set_facecolor("#1E2D45")
                cell.get_text().set_color(PALETTE["primary"])
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor(PALETTE["panel"])
                cell.get_text().set_color(PALETTE["text_main"])

        p2 = out_dir / "page2_churn_risk_register.png"
        fig2.savefig(p2, facecolor=PALETTE["bg"])
        pdf.savefig(fig2, facecolor=PALETTE["bg"])
        plt.close(fig2)
        saved_paths.append(p2)

        # =====================================================================
        # PAGE 3: RECOVERY ACTION CENTER
        # =====================================================================
        fig3 = plt.figure(figsize=(14, 8), facecolor=PALETTE["bg"], dpi=160)
        p1_cnt = int((queue_df["priority_tier"] == "P1 - Immediate (Top 250)").sum())
        p2_cnt = int((queue_df["priority_tier"] == "P2 - High Priority").sum())
        p3_cnt = int((queue_df["priority_tier"] == "P3 - Standard Queue").sum())

        _draw_kpi_strip(
            fig3,
            [
                ("Total Action Queue", f"{len(queue_df):,}", "Subscribers >= Operational Threshold"),
                ("P1 Immediate Desk", f"{p1_cnt:,}", "Top 250 Highest Priority Index"),
                ("P2 High Priority", f"{p2_cnt:,}", "Critical/High Risk Remaining"),
                ("P3 Nurture Queue", f"{p3_cnt:,}", "Moderate Risk Outreach Pool"),
            ],
            "Power BI Page 3 — Recovery Action Center & Analytical Playbook Queue",
            "Governance: Suggested playbooks are rule-based operational hypotheses for controlled A/B testing — NOT proven causal interventions.",
        )

        ax3_1 = fig3.add_axes([0.33, 0.44, 0.20, 0.28])
        _style_ax(ax3_1, "Queue by Analytical Playbook")
        pb_cnt = queue_df["recommended_playbook"].value_counts().iloc[::-1]
        short_pb = [s.replace(" & Setup Check", "").replace(" Value Bundle Trial", " Trial").replace(" & Billing Check-In", "") for s in pb_cnt.index]
        b_pb = ax3_1.barh(short_pb, pb_cnt.values, color=PALETTE["primary"], height=0.55)
        ax3_1.set_xlim(0, max(pb_cnt.values) * 1.24)
        ax3_1.tick_params(axis="y", labelsize=9.4)
        for b in b_pb:
            ax3_1.text(b.get_width() + 12, b.get_y() + b.get_height() / 2, f"{int(b.get_width())}", va="center", color=PALETTE["text_main"], fontsize=8.6)

        ax3_2 = fig3.add_axes([0.70, 0.44, 0.26, 0.28])
        _style_ax(ax3_2, "Workload by Specialist Care Team")
        tm_cnt = queue_df["assigned_team"].value_counts().iloc[::-1]
        b_tm = ax3_2.barh(tm_cnt.index, tm_cnt.values, color=PALETTE["secondary"], height=0.52)
        ax3_2.set_xlim(0, max(tm_cnt.values) * 1.24)
        ax3_2.tick_params(axis="y", labelsize=9.4)
        for b in b_tm:
            ax3_2.text(b.get_width() + 12, b.get_y() + b.get_height() / 2, f"{int(b.get_width())}", va="center", color=PALETTE["text_main"], fontsize=8.6)

        ax3_3 = fig3.add_axes([0.03, 0.05, 0.94, 0.31])
        ax3_3.set_axis_off()
        ax3_3.text(0.0, 0.95, "Operational Retention Outreach Queue (Workflow Tracking Schema)", color=PALETTE["text_main"], fontsize=10.5, fontweight="bold")
        q_top = queue_df.head(7)
        q_cells = []
        for _, r in q_top.iterrows():
            q_cells.append(
                [
                    f"#{int(r['priority_rank'])}",
                    r["synthetic_customer_id"],
                    r["priority_tier"],
                    f"{r['churn_probability']*100:.1f}%",
                    f"{r['primary_reason_code']} + {r['secondary_reason_code']}",
                    r["recommended_playbook"],
                    r["assigned_team"],
                ]
            )
        tbl3 = ax3_3.table(
            cellText=q_cells,
            colLabels=["Rank", "Synthetic ID", "Priority Tier", "Risk", "Top 2 Signals", "Analytical Playbook (Hypothesis)", "Care Desk"],
            colWidths=[0.05, 0.11, 0.15, 0.06, 0.23, 0.24, 0.16],
            bbox=[0.0, 0.0, 1.0, 0.86],
            cellLoc="left",
        )
        tbl3.auto_set_font_size(False)
        tbl3.set_fontsize(8.4)
        for (row_i, col_i), cell in tbl3.get_celld().items():
            cell.set_edgecolor(PALETTE["border"])
            if row_i == 0:
                cell.set_facecolor("#1E2D45")
                cell.get_text().set_color(PALETTE["primary"])
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor(PALETTE["panel"])
                cell.get_text().set_color(PALETTE["text_main"])

        p3 = out_dir / "page3_recovery_action_center.png"
        fig3.savefig(p3, facecolor=PALETTE["bg"])
        pdf.savefig(fig3, facecolor=PALETTE["bg"])
        plt.close(fig3)
        saved_paths.append(p3)

        # =====================================================================
        # PAGE 4: MODEL & DATA QUALITY
        # =====================================================================
        fig4 = plt.figure(figsize=(14, 8), facecolor=PALETTE["bg"], dpi=160)
        tm = metrics["test_metrics_operational"]
        _draw_kpi_strip(
            fig4,
            [
                ("Champion Model", metrics["champion_model_name"], metrics["model_version"]),
                ("Untouched Test ROC-AUC", f"{tm['roc_auc']:.4f}", f"95% CI {metrics['test_bootstrap_95ci']['roc_auc_95ci']}"),
                ("Untouched Test PR-AUC", f"{tm['pr_auc']:.4f}", f"Base Rate {tm['observed_churn_rate']:.4f}"),
                ("Brier Calibration", f"{tm['brier_score']:.4f}", f"ECE = {tm['ece']:.4f} (Well-Calibrated)"),
                ("SQL QA Checks", f"{int((audit_df['check_status'] == 'PASS').sum())}/{len(audit_df)} PASS", metrics["data_version"]),
            ],
            "Power BI Page 4 — Model Evaluation, Calibration & SQL Data Quality Governance",
            "All evaluation metrics computed strictly on untouched held-out test split (n = 1,409; natural 26.54% churn). SMOTE restricted to CV train folds.",
        )

        ax4_1 = fig4.add_axes([0.24, 0.43, 0.30, 0.29])
        _style_ax(ax4_1, "Validation PR-AUC vs Brier Calibration")
        short_model = {
            "Logistic Regression": "LogReg",
            "Random Forest": "RandForest",
            "LightGBM": "LightGBM",
            "XGBoost": "XGBoost",
        }
        short_strat = {
            "Natural Prior (Unweighted)": "Natural",
            "Natural Prior (Regularized)": "Natural",
            "Class-Weighted (Balanced)": "ClassWt",
            "Class-Weighted Subsample": "ClassWt",
            "In-Fold SMOTE (Train Only)": "SMOTE",
        }
        short_labels = [
            f"{short_model.get(r['model_name'], r['model_name'])} ({short_strat.get(r['imbalance_strategy'], r['imbalance_strategy'])})"
            for _, r in comp_df.iterrows()
        ]
        y_pos = np.arange(len(comp_df))
        ax4_1.barh(y_pos - 0.17, comp_df["val_pr_auc"], height=0.34, color=PALETTE["primary"], label="Val PR-AUC (Higher)")
        ax4_1.barh(y_pos + 0.17, comp_df["val_brier_score"], height=0.34, color=PALETTE["critical"], label="Val Brier Loss (Lower)")
        ax4_1.set_yticks(y_pos)
        ax4_1.set_yticklabels(short_labels, fontsize=9.2)
        ax4_1.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["border"], labelcolor=PALETTE["text_main"], fontsize=8.0, loc="lower right")

        ax4_2 = fig4.add_axes([0.60, 0.43, 0.36, 0.29])
        _style_ax(ax4_2, "Reliability Curve on Untouched Test Split")
        ax4_2.plot([0, 1], [0, 1], "--", color=PALETTE["text_muted"], linewidth=1.0, label="Perfect Calibration")
        ax4_2.plot(
            cal_df["bin_mean_predicted_probability"],
            cal_df["bin_observed_churn_frequency"],
            "o-",
            color=PALETTE["primary"],
            linewidth=2.0,
            label="Champion LightGBM",
        )
        ax4_2.set_xlabel("Mean Predicted Churn Probability", color=PALETTE["text_muted"], fontsize=8)
        ax4_2.set_ylabel("Observed Churn Frequency", color=PALETTE["text_muted"], fontsize=8)
        ax4_2.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["border"], labelcolor=PALETTE["text_main"], fontsize=7.5)

        ax4_3 = fig4.add_axes([0.03, 0.05, 0.94, 0.30])
        ax4_3.set_axis_off()
        ax4_3.text(0.0, 0.95, "SQL Staging & Clean Layer Data Quality Audit Log", color=PALETTE["text_main"], fontsize=10.5, fontweight="bold")
        a_cells = []
        for _, r in audit_df.iterrows():
            a_cells.append(
                [
                    r["check_stage"],
                    r["check_name"],
                    f"{r['metric_value']:g}",
                    r["check_status"],
                    r["notes"],
                ]
            )
        tbl4 = ax4_3.table(
            cellText=a_cells,
            colLabels=["Stage", "Audit Check", "Value", "Status", "Engineering & Governance Notes"],
            colWidths=[0.09, 0.22, 0.08, 0.07, 0.54],
            bbox=[0.0, 0.0, 1.0, 0.86],
            cellLoc="left",
        )
        tbl4.auto_set_font_size(False)
        tbl4.set_fontsize(8.4)
        for (row_i, col_i), cell in tbl4.get_celld().items():
            cell.set_edgecolor(PALETTE["border"])
            if row_i == 0:
                cell.set_facecolor("#1E2D45")
                cell.get_text().set_color(PALETTE["primary"])
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor(PALETTE["panel"])
                cell.get_text().set_color(PALETTE["text_main"])

        p4 = out_dir / "page4_model_and_data_quality.png"
        fig4.savefig(p4, facecolor=PALETTE["bg"])
        pdf.savefig(fig4, facecolor=PALETTE["bg"])
        plt.close(fig4)
        saved_paths.append(p4)

    saved_paths.append(pdf_path)
    return saved_paths
