"""Tests for stratified split integrity, untouched test natural ratio, in-fold SMOTE isolation, and calibration."""

from __future__ import annotations

import json
import sqlite3

import pandas as pd
from imblearn.over_sampling import SMOTE

from src.config import METRICS_REPORT_PATH, MODEL_COMPARISON_CSV_PATH, WAREHOUSE_DB_PATH
from src.modeling import get_candidate_pipelines


def test_splits_preserve_untouched_natural_churn_rate() -> None:
    with METRICS_REPORT_PATH.open("r", encoding="utf-8") as fp:
        report = json.load(fp)

    sizes = report["split_sizes"]
    assert sizes["train"] + sizes["validation"] + sizes["test"] == 7043
    assert sizes["test"] == 1409
    assert sizes["validation"] == 1409

    rates = report["natural_churn_rates"]
    # Natural churn is ~0.2654; untouched validation and test must NOT be 0.50 balanced!
    assert abs(rates["test"] - 0.2654) < 0.005
    assert abs(rates["validation"] - 0.2654) < 0.005

    with sqlite3.connect(WAREHOUSE_DB_PATH) as conn:
        scores_df = pd.read_sql("SELECT dataset_split, churn_probability FROM ml_customer_churn_scores", conn)

    assert len(scores_df) == 7043
    assert ((scores_df["churn_probability"] >= 0.0) & (scores_df["churn_probability"] <= 1.0)).all()


def test_smote_isolated_inside_imblearn_pipelines() -> None:
    candidates = get_candidate_pipelines(random_state=42)
    smote_pipes = [pipe for _, strat, pipe in candidates if "SMOTE" in strat]
    assert len(smote_pipes) >= 3
    for pipe in smote_pipes:
        assert "smote" in pipe.named_steps
        assert isinstance(pipe.named_steps["smote"], SMOTE)


def test_champion_metrics_and_calibration_bounds() -> None:
    with METRICS_REPORT_PATH.open("r", encoding="utf-8") as fp:
        report = json.load(fp)

    tm_op = report["test_metrics_operational"]
    tm_05 = report["test_metrics_default_050"]

    assert 0.80 <= tm_op["roc_auc"] <= 0.92
    assert 0.58 <= tm_op["pr_auc"] <= 0.80
    assert tm_op["brier_score"] <= 0.155
    assert tm_op["ece"] <= 0.05
    # Operational threshold must be capacity/cost optimized (distinct from default 0.50) and improve recall
    assert report["operational_threshold"] != 0.50
    assert tm_op["recall"] > tm_05["recall"]
    assert tm_op["flagged_rate"] <= 0.31

    comp_df = pd.read_csv(MODEL_COMPARISON_CSV_PATH)
    assert len(comp_df) >= 8
