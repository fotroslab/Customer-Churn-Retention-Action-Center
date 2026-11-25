#!/usr/bin/env python3
"""End-to-end reproducible pipeline: Raw CSV -> SQL Warehouse -> ML/SHAP -> Power BI Star Schema & Snapshots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion import persist_scoring_and_export_star_schema, run_sql_staging_and_cleaning
from src.modeling import train_evaluate_and_score_all
from src.report_snapshots import generate_all_powerbi_snapshots


def main() -> None:
    print("[1/4] Running SQL staging, data cleaning, pseudonymization & quality checks...")
    sql_summary = run_sql_staging_and_cleaning()
    print(
        f"      Cleaned rows: {sql_summary['clean_rows']:,} | "
        f"Zero-tenure blank TotalCharges handled: {sql_summary['zero_tenure_imputed']} | "
        f"Natural churn rate: {sql_summary['natural_churn_rate']*100:.2f}%"
    )

    print("[2/4] Training & comparing models (Natural vs Class-Weight vs In-Fold SMOTE) + SHAP...")
    ml_out = train_evaluate_and_score_all()
    report = ml_out["metrics_report"]
    tm_op = report["test_metrics_operational"]
    print(
        f"      Selected Champion: {report['champion_key']} | "
        f"Operational Threshold (Val-optimized): {report['operational_threshold']:.2f}"
    )
    print(
        f"      Untouched Test Metrics -> ROC-AUC: {tm_op['roc_auc']:.4f} | "
        f"PR-AUC: {tm_op['pr_auc']:.4f} | Brier: {tm_op['brier_score']:.4f} | "
        f"Recall: {tm_op['recall']:.4f} | Precision: {tm_op['precision']:.4f} | F1: {tm_op['f1']:.4f}"
    )

    print("[3/4] Writing scoring & action queue tables + exporting Power BI Star Schema CSVs...")
    mart_counts = persist_scoring_and_export_star_schema(
        scores_df=ml_out["scores_df"],
        action_queue_df=ml_out["action_queue_df"],
        eval_registry_df=ml_out["eval_registry_df"],
    )
    print("      Exported Star Schema tables:", json.dumps(mart_counts))

    print("[4/4] Rendering 4-page Power BI reference snapshots (PNG + PDF)...")
    snap_paths = generate_all_powerbi_snapshots()
    print(f"      Generated {len(snap_paths)} snapshot files in powerbi/snapshots/.")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
