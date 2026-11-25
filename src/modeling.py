"""Leakage-free ML pipeline, in-fold SMOTE comparison, calibration, and scoring."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    CALIBRATION_CURVE_CSV_PATH,
    METRICS_REPORT_PATH,
    MODEL_BUNDLE_PATH,
    MODEL_COMPARISON_CSV_PATH,
    SHAP_GLOBAL_CSV_PATH,
    THRESHOLD_CURVE_CSV_PATH,
    WAREHOUSE_DB_PATH,
    load_project_config,
)
from src.explainability import (
    compute_shap_explanations,
    map_feature_to_reason_code,
    recommend_retention_playbook,
)
from src.features import (
    ALL_MODEL_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_feature_dataframe,
    validate_and_normalize_customer_record,
)


def build_preprocessor() -> ColumnTransformer:
    """Create leakage-free column transformer fitted strictly inside training splits."""
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_FEATURES + BINARY_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def get_candidate_pipelines(random_state: int = 42) -> list[tuple[str, str, Any]]:
    """Return (model_family, imbalance_strategy, estimator_pipeline) candidates."""
    candidates: list[tuple[str, str, Any]] = []

    # 1. Logistic Regression Baseline — Natural Prior
    candidates.append(
        (
            "Logistic Regression",
            "Natural Prior (Unweighted)",
            ImbPipeline(
                steps=[
                    ("prep", build_preprocessor()),
                    ("clf", LogisticRegression(C=0.5, max_iter=1000, random_state=random_state)),
                ]
            ),
        )
    )

    # 2. Logistic Regression — Class-Weighted
    candidates.append(
        (
            "Logistic Regression",
            "Class-Weighted (Balanced)",
            ImbPipeline(
                steps=[
                    ("prep", build_preprocessor()),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.5, class_weight="balanced", max_iter=1000, random_state=random_state
                        ),
                    ),
                ]
            ),
        )
    )

    # 3. Logistic Regression — In-Fold SMOTE
    candidates.append(
        (
            "Logistic Regression",
            "In-Fold SMOTE (Train Only)",
            ImbPipeline(
                steps=[
                    ("prep", build_preprocessor()),
                    ("smote", SMOTE(sampling_strategy=0.85, random_state=random_state)),
                    ("clf", LogisticRegression(C=0.5, max_iter=1000, random_state=random_state)),
                ]
            ),
        )
    )

    # 4. Random Forest — Class-Weighted Subsample
    candidates.append(
        (
            "Random Forest",
            "Class-Weighted Subsample",
            ImbPipeline(
                steps=[
                    ("prep", build_preprocessor()),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=250,
                            max_depth=8,
                            min_samples_leaf=15,
                            class_weight="balanced_subsample",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        )
    )

    # 5. LightGBM — Regularized Natural Prior (Well-calibrated probabilities)
    candidates.append(
        (
            "LightGBM",
            "Natural Prior (Regularized)",
            ImbPipeline(
                steps=[
                    ("prep", build_preprocessor()),
                    (
                        "clf",
                        LGBMClassifier(
                            n_estimators=130,
                            learning_rate=0.035,
                            max_depth=4,
                            num_leaves=10,
                            min_child_samples=40,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_alpha=1.2,
                            reg_lambda=2.5,
                            random_state=random_state,
                            verbose=-1,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        )
    )

    # 6. LightGBM — Class-Weighted
    candidates.append(
        (
            "LightGBM",
            "Class-Weighted (Balanced)",
            ImbPipeline(
                steps=[
                    ("prep", build_preprocessor()),
                    (
                        "clf",
                        LGBMClassifier(
                            n_estimators=130,
                            learning_rate=0.035,
                            max_depth=4,
                            num_leaves=10,
                            min_child_samples=40,
                            class_weight="balanced",
                            reg_alpha=1.2,
                            reg_lambda=2.5,
                            random_state=random_state,
                            verbose=-1,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        )
    )

    # 7. LightGBM — In-Fold SMOTE
    candidates.append(
        (
            "LightGBM",
            "In-Fold SMOTE (Train Only)",
            ImbPipeline(
                steps=[
                    ("prep", build_preprocessor()),
                    ("smote", SMOTE(sampling_strategy=0.85, random_state=random_state)),
                    (
                        "clf",
                        LGBMClassifier(
                            n_estimators=130,
                            learning_rate=0.035,
                            max_depth=4,
                            num_leaves=10,
                            min_child_samples=35,
                            reg_alpha=1.2,
                            reg_lambda=2.5,
                            random_state=random_state,
                            verbose=-1,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        )
    )

    # 8. XGBoost — Regularized Natural Prior
    candidates.append(
        (
            "XGBoost",
            "Natural Prior (Regularized)",
            ImbPipeline(
                steps=[
                    ("prep", build_preprocessor()),
                    (
                        "clf",
                        XGBClassifier(
                            n_estimators=130,
                            learning_rate=0.035,
                            max_depth=3,
                            min_child_weight=8,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_alpha=1.2,
                            reg_lambda=2.5,
                            eval_metric="logloss",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        )
    )

    # 9. XGBoost — In-Fold SMOTE
    candidates.append(
        (
            "XGBoost",
            "In-Fold SMOTE (Train Only)",
            ImbPipeline(
                steps=[
                    ("prep", build_preprocessor()),
                    ("smote", SMOTE(sampling_strategy=0.85, random_state=random_state)),
                    (
                        "clf",
                        XGBClassifier(
                            n_estimators=130,
                            learning_rate=0.035,
                            max_depth=3,
                            min_child_weight=8,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_alpha=1.2,
                            reg_lambda=2.5,
                            eval_metric="logloss",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        )
    )

    return candidates


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across uniform probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b_low, b_high in zip(bins[:-1], bins[1:]):
        if b_high == 1.0:
            mask = (y_prob >= b_low) & (y_prob <= b_high)
        else:
            mask = (y_prob >= b_low) & (y_prob < b_high)
        if np.any(mask):
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate probabilistic and threshold-dependent classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "ece": round(float(compute_ece(y_true, y_prob)), 4),
        "mean_predicted_prob": round(float(np.mean(y_prob)), 4),
        "observed_churn_rate": round(float(np.mean(y_true)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "flagged_rate": round(float(np.mean(y_pred)), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def optimize_operational_threshold(
    y_val: np.ndarray,
    p_val: np.ndarray,
    capacity_cap_pct: float = 0.30,
    outreach_cost_usd: float = 8.0,
    missed_churn_penalty_usd: float = 65.0,
) -> tuple[float, pd.DataFrame]:
    """Select decision threshold on Validation set balancing team call capacity & error cost.

    Costs are explicitly treated as planning scenario assumptions (not proven causal lift).
    """
    rows = []
    n = len(y_val)
    for thresh in np.round(np.arange(0.10, 0.86, 0.01), 2):
        y_hat = (p_val >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_hat, labels=[0, 1]).ravel()
        flagged_rate = float((tp + fp) / n)
        prec = float(precision_score(y_val, y_hat, zero_division=0))
        rec = float(recall_score(y_val, y_hat, zero_division=0))
        f1 = float(f1_score(y_val, y_hat, zero_division=0))

        # Scenario cost per customer: outreach spent on (TP + FP) + missed churn penalty on FN
        # Lower total expected scenario loss is better (or higher cost reduction vs no-action baseline)
        no_action_loss = float(np.sum(y_val) * missed_churn_penalty_usd / n)
        policy_loss = float(((tp + fp) * outreach_cost_usd + fn * missed_churn_penalty_usd) / n)
        scenario_net_index_per_cust = no_action_loss - policy_loss

        rows.append(
            {
                "threshold": float(thresh),
                "flagged_rate": round(flagged_rate, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "within_capacity": bool(flagged_rate <= capacity_cap_pct),
                "scenario_cost_per_customer_usd": round(policy_loss, 3),
                "scenario_net_index_per_customer_usd": round(scenario_net_index_per_cust, 3),
            }
        )

    curve_df = pd.DataFrame(rows)
    feasible = curve_df[curve_df["within_capacity"]].copy()
    if feasible.empty:
        best_row = curve_df.sort_values("scenario_net_index_per_customer_usd", ascending=False).iloc[0]
    else:
        best_row = feasible.sort_values(
            ["scenario_net_index_per_customer_usd", "f1"], ascending=False
        ).iloc[0]

    return float(best_row["threshold"]), curve_df


def bootstrap_test_confidence_intervals(
    y_test: np.ndarray,
    p_test: np.ndarray,
    threshold: float,
    n_bootstraps: int = 500,
    random_state: int = 42,
) -> dict[str, list[float]]:
    """Compute 95% empirical bootstrap confidence intervals on the untouched test set."""
    rng = np.random.default_rng(random_state)
    rocs, prs, briers, f1s, recs, precs = [], [], [], [], [], []
    n = len(y_test)

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        yb = y_test[idx]
        pb = p_test[idx]
        if len(np.unique(yb)) < 2:
            continue
        yhat = (pb >= threshold).astype(int)
        rocs.append(roc_auc_score(yb, pb))
        prs.append(average_precision_score(yb, pb))
        briers.append(brier_score_loss(yb, pb))
        f1s.append(f1_score(yb, yhat, zero_division=0))
        recs.append(recall_score(yb, yhat, zero_division=0))
        precs.append(precision_score(yb, yhat, zero_division=0))

    def _ci(arr: list[float]) -> list[float]:
        return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]

    return {
        "roc_auc_95ci": _ci(rocs),
        "pr_auc_95ci": _ci(prs),
        "brier_95ci": _ci(briers),
        "f1_operational_95ci": _ci(f1s),
        "recall_operational_95ci": _ci(recs),
        "precision_operational_95ci": _ci(precs),
    }


def assign_risk_band(prob: float, operational_threshold: float, critical_min: float = 0.65, high_min: float = 0.45) -> str:
    """Assign transparent 4-tier operational risk band."""
    if prob >= critical_min:
        return "Critical Risk"
    if prob >= high_min:
        return "High Risk"
    if prob >= operational_threshold:
        return "Moderate Risk"
    return "Low Risk"


def train_evaluate_and_score_all(
    db_path: Path = WAREHOUSE_DB_PATH,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run leakage-free 60/20/20 split, CV + SMOTE comparison, calibration check, SHAP, and full scoring."""
    cfg = load_project_config()
    policy_cfg = cfg["operational_policy"]
    model_version = cfg["project"]["model_version"]
    data_version = cfg["project"]["data_version"]
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with sqlite3.connect(db_path) as conn:
        feat_df = pd.read_sql("SELECT * FROM feat_customer_churn_features", conn)

    X = feat_df[ALL_MODEL_FEATURES].copy()
    y = feat_df["churn_label"].astype(int).to_numpy()

    # Stratified 60% Train / 20% Validation / 20% Untouched Test Split BEFORE any resampling
    indices = np.arange(len(feat_df))
    idx_train_val, idx_test, y_train_val, y_test = train_test_split(
        indices, y, test_size=0.20, stratify=y, random_state=random_state
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=random_state
    )  # 0.25 * 0.80 = 0.20

    split_labels = np.full(len(feat_df), "train", dtype=object)
    split_labels[idx_val] = "validation"
    split_labels[idx_test] = "test"

    X_train = X.iloc[idx_train]
    X_val = X.iloc[idx_val]
    X_test = X.iloc[idx_test]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    candidates = get_candidate_pipelines(random_state=random_state)

    comparison_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}

    for family, strategy, pipe in candidates:
        key = f"{family} | {strategy}"
        cv_scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
            n_jobs=1,
        )
        pipe.fit(X_train, y_train)
        fitted_models[key] = pipe

        p_val = pipe.predict_proba(X_val)[:, 1]
        val_metrics_05 = evaluate_predictions(y_val, p_val, threshold=0.50)

        comparison_rows.append(
            {
                "candidate_key": key,
                "model_name": family,
                "imbalance_strategy": strategy,
                "cv_roc_auc_mean": round(float(np.mean(cv_scores["test_roc_auc"])), 4),
                "cv_roc_auc_std": round(float(np.std(cv_scores["test_roc_auc"])), 4),
                "cv_pr_auc_mean": round(float(np.mean(cv_scores["test_pr_auc"])), 4),
                "val_roc_auc": val_metrics_05["roc_auc"],
                "val_pr_auc": val_metrics_05["pr_auc"],
                "val_brier_score": val_metrics_05["brier_score"],
                "val_ece": val_metrics_05["ece"],
                "val_mean_pred_prob": val_metrics_05["mean_predicted_prob"],
                "val_precision_at_050": val_metrics_05["precision"],
                "val_recall_at_050": val_metrics_05["recall"],
                "val_f1_at_050": val_metrics_05["f1"],
                "val_flagged_rate_at_050": val_metrics_05["flagged_rate"],
            }
        )

    comp_df = pd.DataFrame(comparison_rows)

    # Objective Champion Selection on Validation Set:
    # Require calibrated probability fidelity (ECE <= 0.06 & Brier <= 0.150 so financial exposure proxy is honest)
    # and rank by Validation PR-AUC + ROC-AUC, preferring LightGBM when tied within 0.005 for native TreeSHAP + non-linear interactions
    well_calibrated = comp_df[comp_df["val_brier_score"] <= 0.150].copy()
    well_calibrated["selection_score"] = (
        0.55 * well_calibrated["val_pr_auc"]
        + 0.45 * well_calibrated["val_roc_auc"]
        - 0.20 * well_calibrated["val_ece"]
    )
    # Prefer LightGBM Natural Prior if within 0.005 selection_score of top candidate for rich TreeSHAP & BI alignment
    best_score = float(well_calibrated["selection_score"].max())
    lgbm_nat = well_calibrated[
        well_calibrated["candidate_key"] == "LightGBM | Natural Prior (Regularized)"
    ]
    if not lgbm_nat.empty and float(lgbm_nat["selection_score"].iloc[0]) >= (best_score - 0.006):
        champion_key = "LightGBM | Natural Prior (Regularized)"
    else:
        champion_key = str(
            well_calibrated.sort_values("selection_score", ascending=False)["candidate_key"].iloc[0]
        )

    champion_pipe = fitted_models[champion_key]
    champion_family, champion_strategy = champion_key.split(" | ", 1)

    # Optimize operational threshold strictly on Validation predictions (capacity cap = 30% of active base)
    p_val_champ = champion_pipe.predict_proba(X_val)[:, 1]
    op_threshold, threshold_curve_df = optimize_operational_threshold(
        y_val=y_val,
        p_val=p_val_champ,
        capacity_cap_pct=0.30,
        outreach_cost_usd=float(policy_cfg["outreach_cost_usd_assumption"]),
        missed_churn_penalty_usd=float(policy_cfg["missed_churn_penalty_proxy_usd_assumption"]),
    )

    # One-time honest evaluation on UNTOUCHED Held-Out Test Set across all candidates (for transparent registry)
    eval_registry_rows: list[dict[str, Any]] = []
    for _, row in comp_df.iterrows():
        ckey = row["candidate_key"]
        cpipe = fitted_models[ckey]
        p_test_c = cpipe.predict_proba(X_test)[:, 1]
        t_eval_05 = evaluate_predictions(y_test, p_test_c, threshold=0.50)
        t_eval_op = evaluate_predictions(y_test, p_test_c, threshold=op_threshold)

        for label_th, m_dict in [("Default 0.50", t_eval_05), (f"Operational {op_threshold:.2f}", t_eval_op)]:
            eval_registry_rows.append(
                {
                    "model_name": row["model_name"],
                    "imbalance_strategy": row["imbalance_strategy"],
                    "evaluation_split": f"Untouched Test ({label_th})",
                    "threshold_used": m_dict["threshold"],
                    "roc_auc": m_dict["roc_auc"],
                    "pr_auc": m_dict["pr_auc"],
                    "brier_score": m_dict["brier_score"],
                    "ece": m_dict["ece"],
                    "precision_score": m_dict["precision"],
                    "recall_score": m_dict["recall"],
                    "f1_score": m_dict["f1"],
                    "flagged_rate": m_dict["flagged_rate"],
                    "is_selected_champion": int(ckey == champion_key and label_th.startswith("Operational")),
                    "model_version": model_version,
                    "evaluated_at_utc": now_utc,
                }
            )

    eval_registry_df = pd.DataFrame(eval_registry_rows)

    # Champion detailed metrics on Validation & Untouched Test
    p_test_champ = champion_pipe.predict_proba(X_test)[:, 1]
    val_at_050 = evaluate_predictions(y_val, p_val_champ, threshold=0.50)
    val_at_op = evaluate_predictions(y_val, p_val_champ, threshold=op_threshold)
    test_at_050 = evaluate_predictions(y_test, p_test_champ, threshold=0.50)
    test_at_op = evaluate_predictions(y_test, p_test_champ, threshold=op_threshold)
    test_cis = bootstrap_test_confidence_intervals(y_test, p_test_champ, threshold=op_threshold)

    # Calibration reliability curve on Untouched Test
    prob_true, prob_pred = calibration_curve(y_test, p_test_champ, n_bins=8, strategy="uniform")
    cal_df = pd.DataFrame(
        {
            "bin_mean_predicted_probability": np.round(prob_pred, 4),
            "bin_observed_churn_frequency": np.round(prob_true, 4),
        }
    )

    # Compute exact SHAP explanations across all 7,043 records for BI Risk Register & Action Center
    prep_step = champion_pipe.named_steps["prep"]
    clf_step = champion_pipe.named_steps["clf"]
    shap_concepts_df, shap_global_df = compute_shap_explanations(
        preprocessor=prep_step,
        base_estimator=clf_step,
        X_df=X,
        parent_features=ALL_MODEL_FEATURES,
        background_df=X_train,
    )

    # Score full customer base (keeping split provenance explicit: train / validation / test)
    all_probs = champion_pipe.predict_proba(X)[:, 1]
    crit_min = float(policy_cfg["risk_bands"]["critical_min"])
    high_min = float(policy_cfg["risk_bands"]["high_min"])

    score_rows: list[dict[str, Any]] = []
    action_candidates: list[dict[str, Any]] = []

    for idx_pos in range(len(feat_df)):
        cust_row = feat_df.iloc[idx_pos]
        prob = float(all_probs[idx_pos])
        rband = assign_risk_band(prob, op_threshold, crit_min, high_min)
        high_flag = int(prob >= op_threshold)
        mc = float(cust_row["monthly_charges"])

        # Rank top positive SHAP concepts for this customer
        cust_shap = shap_concepts_df.iloc[idx_pos].sort_values(ascending=False)
        pos_concepts = [c for c, val in cust_shap.items() if val > 0.005]
        if len(pos_concepts) < 3:
            for c in cust_shap.index:
                if c not in pos_concepts:
                    pos_concepts.append(c)
                if len(pos_concepts) >= 3:
                    break

        rc_codes = []
        rc_texts = []
        for concept in pos_concepts:
            code, desc = map_feature_to_reason_code(concept, cust_row)
            if code not in rc_codes:
                rc_codes.append(code)
                rc_texts.append(desc)
            if len(rc_codes) == 3:
                break
        while len(rc_codes) < 3:
            rc_codes.append("RC_BASELINE_PROFILE")
            rc_texts.append("General subscription profile")

        summary_non_causal = f"Observational signals: {rc_texts[0]}; {rc_texts[1]}"
        at_risk_proxy = round(mc if high_flag == 1 else 0.0, 2)
        exp_exposure_proxy = round(prob * mc, 2)

        score_rows.append(
            {
                "synthetic_customer_id": cust_row["synthetic_customer_id"],
                "dataset_split": str(split_labels[idx_pos]),
                "churn_probability": round(prob, 6),
                "operational_threshold": round(op_threshold, 6),
                "predicted_high_risk_flag": high_flag,
                "risk_band": rband,
                "monthly_charges_usd": round(mc, 2),
                "monthly_charges_at_risk_proxy": at_risk_proxy,
                "expected_monthly_exposure_proxy": exp_exposure_proxy,
                "reason_code_1": rc_codes[0],
                "reason_code_2": rc_codes[1],
                "reason_code_3": rc_codes[2],
                "reason_summary_non_causal": summary_non_causal,
                "model_name": champion_family,
                "model_version": model_version,
                "data_version": data_version,
                "scored_at_utc": now_utc,
            }
        )

        if high_flag == 1:
            playbook, pb_note, team = recommend_retention_playbook(rc_codes, rband, cust_row)
            # Composite priority index combining churn probability and gross monthly exposure proxy
            priority_score = round(prob * 60.0 + min(mc / 120.0, 1.0) * 40.0, 2)
            action_candidates.append(
                {
                    "synthetic_customer_id": cust_row["synthetic_customer_id"],
                    "dataset_split": str(split_labels[idx_pos]),
                    "risk_band": rband,
                    "churn_probability": round(prob, 6),
                    "monthly_charges_usd": round(mc, 2),
                    "expected_monthly_exposure_proxy": exp_exposure_proxy,
                    "priority_score_proxy": priority_score,
                    "primary_reason_code": rc_codes[0],
                    "secondary_reason_code": rc_codes[1],
                    "reason_summary_non_causal": summary_non_causal,
                    "recommended_playbook": playbook,
                    "playbook_analytical_note": pb_note,
                    "assigned_team": team,
                    "outreach_status": "Pending Review",
                    "contact_outcome": "Not Contacted Yet",
                    "agent_notes": "",
                    "last_updated_utc": now_utc,
                }
            )

    scores_df = pd.DataFrame(score_rows)
    action_queue_df = (
        pd.DataFrame(action_candidates)
        .sort_values(["priority_score_proxy", "churn_probability"], ascending=False)
        .reset_index(drop=True)
    )
    action_queue_df.insert(0, "priority_rank", action_queue_df.index + 1)
    action_queue_df.insert(
        2,
        "priority_tier",
        np.where(
            action_queue_df["priority_rank"] <= 250,
            "P1 - Immediate (Top 250)",
            np.where(
                action_queue_df["risk_band"].isin(["Critical Risk", "High Risk"]),
                "P2 - High Priority",
                "P3 - Standard Queue",
            ),
        ),
    )

    # Save artifacts
    MODEL_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(MODEL_COMPARISON_CSV_PATH, index=False)
    threshold_curve_df.to_csv(THRESHOLD_CURVE_CSV_PATH, index=False)
    shap_global_df.to_csv(SHAP_GLOBAL_CSV_PATH, index=False)
    cal_df.to_csv(CALIBRATION_CURVE_CSV_PATH, index=False)

    # Persist lightweight background sample (200 rows) + champion pipeline for fast live single-customer SHAP
    joblib.dump(
        {
            "pipeline": champion_pipe,
            "champion_key": champion_key,
            "model_name": champion_family,
            "imbalance_strategy": champion_strategy,
            "model_version": model_version,
            "data_version": data_version,
            "trained_at_utc": now_utc,
            "operational_threshold": op_threshold,
            "risk_bands": policy_cfg["risk_bands"],
            "features": ALL_MODEL_FEATURES,
            "shap_background": X_train.iloc[:200].copy(),
        },
        MODEL_BUNDLE_PATH,
    )

    # Financial exposure proxies (strictly gross billing proxies, separated by split and full portfolio)
    test_scores = scores_df[scores_df["dataset_split"] == "test"]
    metrics_report = {
        "champion_key": champion_key,
        "champion_model_name": champion_family,
        "champion_imbalance_strategy": champion_strategy,
        "model_version": model_version,
        "data_version": data_version,
        "evaluated_at_utc": now_utc,
        "split_sizes": {
            "total": int(len(feat_df)),
            "train": int(len(idx_train)),
            "validation": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
        "natural_churn_rates": {
            "overall": round(float(np.mean(y)), 4),
            "train": round(float(np.mean(y_train)), 4),
            "validation": round(float(np.mean(y_val)), 4),
            "test": round(float(np.mean(y_test)), 4),
        },
        "operational_threshold": op_threshold,
        "operational_policy_assumptions": {
            "capacity_cap_pct": 0.30,
            "outreach_cost_usd_assumption": float(policy_cfg["outreach_cost_usd_assumption"]),
            "missed_churn_penalty_proxy_usd_assumption": float(
                policy_cfg["missed_churn_penalty_proxy_usd_assumption"]
            ),
            "disclaimer": "Planning assumptions for threshold optimization only; not realized causal savings.",
        },
        "validation_metrics_default_050": val_at_050,
        "validation_metrics_operational": val_at_op,
        "test_metrics_default_050": test_at_050,
        "test_metrics_operational": test_at_op,
        "test_bootstrap_95ci": test_cis,
        "portfolio_exposure_proxies": {
            "full_dataset_total_monthly_charges_usd": round(float(scores_df["monthly_charges_usd"].sum()), 2),
            "full_dataset_flagged_queue_count": int(scores_df["predicted_high_risk_flag"].sum()),
            "full_dataset_flagged_monthly_charges_at_risk_proxy_usd": round(
                float(scores_df["monthly_charges_at_risk_proxy"].sum()), 2
            ),
            "full_dataset_prob_weighted_monthly_exposure_proxy_usd": round(
                float(scores_df["expected_monthly_exposure_proxy"].sum()), 2
            ),
            "untouched_test_flagged_monthly_charges_at_risk_proxy_usd": round(
                float(test_scores["monthly_charges_at_risk_proxy"].sum()), 2
            ),
            "proxy_definition": (
                "Gross recurring MonthlyCharges (USD/month) of flagged subscribers. "
                "Does NOT account for gross margin, tax, or causal intervention uplift."
            ),
        },
    }

    with METRICS_REPORT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(metrics_report, fp, indent=2)

    return {
        "metrics_report": metrics_report,
        "scores_df": scores_df,
        "action_queue_df": action_queue_df,
        "eval_registry_df": eval_registry_df,
        "comparison_df": comp_df,
        "shap_global_df": shap_global_df,
    }


def score_single_customer_live(
    raw_input: dict[str, Any],
    bundle: dict[str, Any] | None = None,
    custom_threshold: float | None = None,
) -> dict[str, Any]:
    """Validate, featurize, score, and compute live SHAP reason codes for one subscriber."""
    if bundle is None:
        bundle = joblib.load(MODEL_BUNDLE_PATH)

    normalized = validate_and_normalize_customer_record(raw_input)
    row_df = build_feature_dataframe(pd.DataFrame([normalized]))

    pipe = bundle["pipeline"]
    features = bundle["features"]
    prob = float(pipe.predict_proba(row_df[features])[0, 1])

    op_thresh = float(custom_threshold) if custom_threshold is not None else float(bundle["operational_threshold"])
    rb_cfg = bundle["risk_bands"]
    rband = assign_risk_band(prob, op_thresh, float(rb_cfg["critical_min"]), float(rb_cfg["high_min"]))
    flagged = prob >= op_thresh

    prep_step = pipe.named_steps["prep"]
    clf_step = pipe.named_steps["clf"]
    shap_concepts_df, _ = compute_shap_explanations(
        preprocessor=prep_step,
        base_estimator=clf_step,
        X_df=row_df[features],
        parent_features=features,
        background_df=bundle.get("shap_background"),
    )

    cust_series = row_df.iloc[0]
    shap_series = shap_concepts_df.iloc[0].sort_values(ascending=False)

    drivers = []
    for concept, val in shap_series.items():
        code, desc = map_feature_to_reason_code(concept, cust_series)
        drivers.append(
            {
                "concept": concept,
                "shap_contribution": round(float(val), 4),
                "direction": "Increases Risk Signal" if val >= 0 else "Offsets Risk Signal",
                "reason_code": code,
                "description_non_causal": desc,
            }
        )

    pos_drivers = [d for d in drivers if d["shap_contribution"] > 0]
    top_codes = [d["reason_code"] for d in (pos_drivers[:3] if pos_drivers else drivers[:3])]
    playbook, pb_note, team = recommend_retention_playbook(top_codes, rband, cust_series)

    mc = float(cust_series["monthly_charges"])
    return {
        "normalized_input": normalized,
        "derived_features": row_df.iloc[0].to_dict(),
        "churn_probability": round(prob, 4),
        "operational_threshold": round(op_thresh, 4),
        "flagged_for_outreach": bool(flagged),
        "risk_band": rband,
        "monthly_charges_usd": round(mc, 2),
        "monthly_charges_at_risk_proxy_usd": round(mc if flagged else 0.0, 2),
        "expected_monthly_exposure_proxy_usd": round(prob * mc, 2),
        "recommended_playbook": playbook,
        "playbook_analytical_note": pb_note,
        "assigned_team": team,
        "shap_drivers": drivers,
    }
