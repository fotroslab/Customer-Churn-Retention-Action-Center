"""Downloadable export helpers (CSV, Excel .xlsx, Executive PDF brief) with safe fallbacks."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def action_pack_to_excel_bytes(
    action_queue_df: pd.DataFrame,
    model_eval_df: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> bytes | None:
    """Build multi-sheet Excel workbook (.xlsx) for Retention Operations & BI audit.

    Returns None gracefully if openpyxl fails so optional export never crashes the UI.
    """
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            action_queue_df.to_excel(writer, sheet_name="Retention_Action_Queue", index=False)
            model_eval_df.to_excel(writer, sheet_name="Model_Evaluation_Test", index=False)
            audit_df.to_excel(writer, sheet_name="SQL_Data_Quality_Audit", index=False)
        return buf.getvalue()
    except Exception:
        return None


def generate_executive_pdf_brief(metrics_report: dict[str, Any], top_queue_df: pd.DataFrame) -> bytes | None:
    """Generate a concise 1-page PDF executive summary of executed metrics & top priority queue.

    Wrapped safely so optional PDF rendering issues never disrupt core application workflows.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=32,
            rightMargin=32,
            topMargin=32,
            bottomMargin=32,
            title="Customer Churn Intelligence — Executive Brief",
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ExecTitle",
            parent=styles["Heading1"],
            fontSize=15,
            textColor=colors.HexColor("#0B1220"),
            spaceAfter=6,
        )
        sub_style = ParagraphStyle(
            "ExecSub",
            parent=styles["Normal"],
            fontSize=9.5,
            textColor=colors.HexColor("#334155"),
            leading=13,
            spaceAfter=10,
        )
        sec_style = ParagraphStyle(
            "ExecSec",
            parent=styles["Heading2"],
            fontSize=11.5,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=8,
            spaceAfter=4,
        )

        story = []
        story.append(Paragraph("Customer Churn Intelligence & Retention Action Center", title_style))
        story.append(
            Paragraph(
                f"Model: {metrics_report['champion_key']} ({metrics_report['model_version']}) | "
                f"Data Version: {metrics_report['data_version']} | "
                f"Evaluated UTC: {metrics_report['evaluated_at_utc']}",
                sub_style,
            )
        )

        tm_op = metrics_report["test_metrics_operational"]
        tm_05 = metrics_report["test_metrics_default_050"]
        exp = metrics_report["portfolio_exposure_proxies"]
        ci = metrics_report["test_bootstrap_95ci"]

        story.append(Paragraph("1. Honest Untouched Test Set Performance (n = 1,409; Natural Churn = 26.54%)", sec_style))
        kpi_data = [
            ["Metric", "Operational Threshold", "Default 0.50", "95% Bootstrap CI (Operational)"],
            ["Threshold Cutoff", f"{tm_op['threshold']:.2f}", "0.50", "Selected on Validation Set"],
            ["ROC-AUC", f"{tm_op['roc_auc']:.4f}", f"{tm_05['roc_auc']:.4f}", f"[{ci['roc_auc_95ci'][0]:.4f}, {ci['roc_auc_95ci'][1]:.4f}]"],
            ["PR-AUC", f"{tm_op['pr_auc']:.4f}", f"{tm_05['pr_auc']:.4f}", f"[{ci['pr_auc_95ci'][0]:.4f}, {ci['pr_auc_95ci'][1]:.4f}]"],
            ["Brier Calibration", f"{tm_op['brier_score']:.4f}", f"{tm_05['brier_score']:.4f}", f"[{ci['brier_95ci'][0]:.4f}, {ci['brier_95ci'][1]:.4f}]"],
            ["Precision", f"{tm_op['precision']:.4f}", f"{tm_05['precision']:.4f}", f"[{ci['precision_operational_95ci'][0]:.4f}, {ci['precision_operational_95ci'][1]:.4f}]"],
            ["Recall (Churn Capture)", f"{tm_op['recall']:.4f}", f"{tm_05['recall']:.4f}", f"[{ci['recall_operational_95ci'][0]:.4f}, {ci['recall_operational_95ci'][1]:.4f}]"],
            ["F1-Score (Natural Test)", f"{tm_op['f1']:.4f}", f"{tm_05['f1']:.4f}", f"[{ci['f1_operational_95ci'][0]:.4f}, {ci['f1_operational_95ci'][1]:.4f}]"],
            ["Outreach Queue Rate", f"{tm_op['flagged_rate']*100:.1f}%", f"{tm_05['flagged_rate']*100:.1f}%", "Within 30% Team Capacity Cap"],
        ]
        t1 = Table(kpi_data, colWidths=[145, 115, 95, 165])
        t1.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#141F30")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t1)
        story.append(Spacer(1, 8))

        story.append(Paragraph("2. Portfolio Exposure Proxy & Governance Guardrails", sec_style))
        story.append(
            Paragraph(
                f"Flagged Retention Queue: <b>{exp['full_dataset_flagged_queue_count']:,}</b> subscribers "
                f"| Gross Monthly Charges in Queue (Proxy): <b>${exp['full_dataset_flagged_monthly_charges_at_risk_proxy_usd']:,.2f}/mo</b> "
                f"| Probability-Weighted Monthly Exposure Proxy: <b>${exp['full_dataset_prob_weighted_monthly_exposure_proxy_usd']:,.2f}/mo</b>.<br/>"
                "<i>Disclaimer:</i> Monthly charges represent gross recurring billing exposure, NOT net margin or causally saved revenue. "
                "SHAP reason codes reflect observational associations, not proven root causes; retention playbooks require controlled A/B uplift validation.",
                sub_style,
            )
        )

        story.append(Paragraph("3. Sample Top-Priority Retention Queue (Synthetic Non-Sensitive IDs)", sec_style))
        q_rows = [["Rank", "Customer ID", "Risk Band", "Churn Prob", "Bill ($/mo)", "Primary Signal", "Suggested Playbook"]]
        for _, r in top_queue_df.head(8).iterrows():
            q_rows.append(
                [
                    str(int(r["priority_rank"])),
                    str(r["synthetic_customer_id"]),
                    str(r["risk_band"]),
                    f"{float(r['churn_probability'])*100:.1f}%",
                    f"${float(r['monthly_charges_usd']):.2f}",
                    str(r["primary_reason_code"]),
                    str(r["recommended_playbook"])[:34],
                ]
            )
        t2 = Table(q_rows, colWidths=[32, 78, 68, 54, 56, 98, 134])
        t2.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#141F30")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                    ("PADDING", (0, 0), (-1, -1), 3.5),
                ]
            )
        )
        story.append(t2)

        doc.build(story)
        return buf.getvalue()
    except Exception:
        return None
