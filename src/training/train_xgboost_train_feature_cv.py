"""Train XGBoost on train_feature data only with 5-fold CV."""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.config import config
from src.data.cleaner import clean_dataframe
from src.data.loader import get_train_path, read_csv
from src.evaluation.metrics import calculate_classification_metrics_with_threshold
from src.features.interaction_features import add_interaction_features
from src.features.main_features import add_main_features
from src.preprocessing.encoder import FoldEncoder
from src.preprocessing.splitter import stratified_holdout_split, stratified_train_validation_split
from src.utils.io import ensure_directory
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _prepare_training_frame(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target vector."""
    if config.TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Required target column not found: {config.TARGET_COLUMN}")
    if config.KEY_COLUMN not in dataset.columns:
        raise ValueError(f"Required key column not found: {config.KEY_COLUMN}")
    y = dataset[config.TARGET_COLUMN].astype(int)
    x = dataset.drop(columns=[config.TARGET_COLUMN, config.KEY_COLUMN], errors="ignore")
    return x, y


def _compute_scale_pos_weight(y_data: pd.Series) -> float:
    """Compute class weight as negative/positive."""
    pos_count = float((y_data == 1).sum())
    neg_count = float((y_data == 0).sum())
    return 1.0 if pos_count == 0 else neg_count / pos_count


def _xgboost_params(scale_pos_weight: float) -> dict[str, Any]:
    """Return default XGBoost parameters."""
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": config.RANDOM_SEED,
        "n_jobs": -1,
        "scale_pos_weight": float(scale_pos_weight),
    }


def _collect_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute summary metrics at a fixed threshold."""
    metrics = calculate_classification_metrics_with_threshold(y_true.to_numpy(), y_score, threshold)
    y_pred = (y_score >= threshold).astype(int)
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0))
    return metrics


def _cross_validate(
    x_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run 5-fold stratified cross-validation on train_feature data."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    fold_metrics: list[dict[str, Any]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(x_train_full, y_train_full), start=1):
        x_train_fold = x_train_full.iloc[train_idx]
        y_train_fold = y_train_full.iloc[train_idx]
        x_valid_fold = x_train_full.iloc[valid_idx]
        y_valid_fold = y_train_full.iloc[valid_idx]

        encoder = FoldEncoder().fit(x_train_fold)
        x_train_encoded = encoder.transform(x_train_fold)
        x_valid_encoded = encoder.transform(x_valid_fold)
        scale_pos_weight = _compute_scale_pos_weight(y_train_fold)
        model = XGBClassifier(**_xgboost_params(scale_pos_weight))
        model.fit(x_train_encoded, y_train_fold)
        valid_score = model.predict_proba(x_valid_encoded)[:, 1]

        metrics = _collect_metrics(y_valid_fold, valid_score, threshold)
        metrics["fold"] = fold_idx
        metrics["scale_pos_weight"] = float(scale_pos_weight)
        fold_metrics.append(metrics)

        logger.info(
            "train_feature XGBoost CV fold=%s auc=%.6f pr_auc=%.6f f1=%.6f",
            fold_idx,
            metrics["auc_roc"],
            metrics["pr_auc"],
            metrics["f1_score"],
        )

    keys = [
        "auc_roc",
        "gini",
        "ks_statistic",
        "pr_auc",
        "brier_score",
        "precision",
        "recall",
        "f1_score",
    ]
    summary: dict[str, float] = {}
    for key in keys:
        values = [fold[key] for fold in fold_metrics]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return fold_metrics, summary


def _train_final_model(
    x_train_full: pd.DataFrame,
    y_train_full: pd.Series,
) -> tuple[XGBClassifier, FoldEncoder, float]:
    """Train final XGBoost model on train_feature train split."""
    x_train, x_valid, y_train, _ = stratified_train_validation_split(x_train_full, y_train_full)
    encoder = FoldEncoder().fit(x_train)
    x_train_encoded = encoder.transform(x_train)
    x_valid_encoded = encoder.transform(x_valid)
    scale_pos_weight = _compute_scale_pos_weight(y_train)
    model = XGBClassifier(**_xgboost_params(scale_pos_weight))
    model.fit(x_train_encoded, y_train)
    # Keep variable usage explicit for future extension.
    _ = x_valid_encoded
    return model, encoder, float(scale_pos_weight)


def _save_model_bundle(
    model: XGBClassifier,
    encoder: FoldEncoder,
    threshold: float,
    trained_at: str,
    scale_pos_weight: float,
) -> Path:
    """Save train_feature-only XGBoost model bundle."""
    ensure_directory(config.MODELS_SAVED_DIR)
    threshold_tag = f"{int(round(threshold * 100)):03d}"
    model_path = config.MODELS_SAVED_DIR / f"xgboost_train_feature_cv5_threshold{threshold_tag}_{trained_at}.pkl"
    payload = {
        "model": model,
        "encoder": encoder,
        "threshold": float(threshold),
        "version": "xgboost_train_feature_cv5",
        "trained_at": trained_at,
        "source": "train_feature_only",
        "scale_pos_weight": float(scale_pos_weight),
    }
    with model_path.open("wb") as output_file:
        pickle.dump(payload, output_file)
    return model_path


def _threshold_table(y_true: pd.Series, y_score: np.ndarray) -> list[dict[str, float]]:
    """Compute precision/recall table for selected thresholds."""
    rows = []
    for threshold in np.arange(0.10, 1.00, 0.10):
        y_pred = (y_score >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return rows


def _save_precision_recall_chart(y_true: pd.Series, y_score: np.ndarray, target_path: Path) -> None:
    """Save precision-recall curve chart."""
    ensure_directory(target_path.parent)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="navy", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("XGBoost Train-Feature Precision-Recall Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(target_path, dpi=120)
    plt.close(fig)


def _write_report(
    threshold: float,
    cv_summary: dict[str, float],
    fold_metrics: list[dict[str, Any]],
    holdout_metrics: dict[str, Any],
    model_path: Path,
    scale_pos_weight: float,
    pr_chart_path: Path,
    threshold_rows: list[dict[str, float]],
) -> Path:
    """Write train_feature-only XGBoost metrics report."""
    ensure_directory(config.REPORTS_DIR)
    report_path = config.REPORTS_DIR / "metrics_xgboost_train_feature_cv5.md"
    lines = [
        "# XGBoost CV5 Metrics (Train Feature Only)",
        "",
        f"Threshold: `{threshold:.2f}`",
        f"Scale pos weight (final train split): `{scale_pos_weight:.6f}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Validation AUC (CV mean +/- std) | {cv_summary['auc_roc_mean']:.6f} +/- {cv_summary['auc_roc_std']:.6f} |",
        f"| Holdout AUC | {holdout_metrics['auc_roc']:.6f} |",
        f"| Gini (CV mean +/- std) | {cv_summary['gini_mean']:.6f} +/- {cv_summary['gini_std']:.6f} |",
        f"| KS (CV mean +/- std) | {cv_summary['ks_statistic_mean']:.6f} +/- {cv_summary['ks_statistic_std']:.6f} |",
        f"| PR-AUC (CV mean +/- std) | {cv_summary['pr_auc_mean']:.6f} +/- {cv_summary['pr_auc_std']:.6f} |",
        f"| Brier Score (CV mean +/- std) | {cv_summary['brier_score_mean']:.6f} +/- {cv_summary['brier_score_std']:.6f} |",
        f"| Precision (CV mean +/- std) | {cv_summary['precision_mean']:.6f} +/- {cv_summary['precision_std']:.6f} |",
        f"| Recall (CV mean +/- std) | {cv_summary['recall_mean']:.6f} +/- {cv_summary['recall_std']:.6f} |",
        f"| F1-Score (CV mean +/- std) | {cv_summary['f1_score_mean']:.6f} +/- {cv_summary['f1_score_std']:.6f} |",
        "",
        "## Holdout Confusion Matrix",
        f"- `{holdout_metrics['confusion_matrix']}`",
        "",
        "## Precision-Recall Chart",
        f"- Chart file: `{pr_chart_path}`",
        "",
        "## Precision-Recall by Threshold (Holdout)",
        "| Threshold | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|",
    ]
    for row in threshold_rows:
        lines.append(
            f"| {row['threshold']:.2f} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} |"
        )
    lines.extend(["", "## Fold Details"])
    for fold in fold_metrics:
        lines.append(
            f"- Fold {fold['fold']}: AUC={fold['auc_roc']:.6f}, PR-AUC={fold['pr_auc']:.6f}, "
            f"Precision={fold['precision']:.6f}, Recall={fold['recall']:.6f}, F1={fold['f1_score']:.6f}"
        )
    lines.extend(["", f"Saved model: `{model_path}`"])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def train_xgboost_train_feature_cv5(threshold: float = 0.30) -> dict[str, Any]:
    """Train XGBoost using train_feature data only, with CV5 and fixed threshold."""
    train_path = get_train_path()
    dataset = read_csv(train_path)
    dataset_clean = clean_dataframe(dataset)
    dataset_featured = add_interaction_features(add_main_features(dataset_clean))

    x, y = _prepare_training_frame(dataset_featured)
    x_train_full, x_holdout, y_train_full, y_holdout = stratified_holdout_split(x, y)
    fold_metrics, cv_summary = _cross_validate(x_train_full, y_train_full, threshold=threshold)

    final_model, final_encoder, final_scale_pos_weight = _train_final_model(x_train_full, y_train_full)
    x_holdout_encoded = final_encoder.transform(x_holdout)
    holdout_score = final_model.predict_proba(x_holdout_encoded)[:, 1]
    holdout_metrics = _collect_metrics(y_holdout, holdout_score, threshold)

    trained_at = datetime.now().strftime(config.MODEL_DATE_FORMAT)
    model_path = _save_model_bundle(
        final_model,
        final_encoder,
        threshold,
        trained_at,
        final_scale_pos_weight,
    )

    pr_chart_path = config.REPORTS_DIR / "xgboost_train_feature_precision_recall_curve.png"
    _save_precision_recall_chart(y_holdout, holdout_score, pr_chart_path)
    threshold_rows = _threshold_table(y_holdout, holdout_score)
    report_path = _write_report(
        threshold,
        cv_summary,
        fold_metrics,
        holdout_metrics,
        model_path,
        final_scale_pos_weight,
        pr_chart_path,
        threshold_rows,
    )

    return {
        "threshold": threshold,
        "model_path": str(model_path),
        "report_path": str(report_path),
        "cv_summary": cv_summary,
        "fold_metrics": fold_metrics,
        "holdout_metrics": holdout_metrics,
        "scale_pos_weight": final_scale_pos_weight,
        "pr_chart_path": str(pr_chart_path),
        "threshold_rows": threshold_rows,
    }
