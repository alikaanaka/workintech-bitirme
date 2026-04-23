"""Train CatBoost with 5-fold CV and fixed-threshold evaluation."""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from src.config import config
from src.evaluation.metrics import calculate_classification_metrics_with_threshold
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


def _catboost_params() -> dict[str, Any]:
    """Return default CatBoost parameters."""
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": config.RANDOM_SEED,
        "allow_writing_files": False,
        "verbose": False,
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
    """Run 5-fold stratified cross-validation on training split."""
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

        model = CatBoostClassifier(**_catboost_params())
        model.fit(x_train_encoded, y_train_fold, eval_set=(x_valid_encoded, y_valid_fold), use_best_model=True)
        valid_score = model.predict_proba(x_valid_encoded)[:, 1]
        metrics = _collect_metrics(y_valid_fold, valid_score, threshold)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

        logger.info(
            "CV fold=%s auc=%.6f pr_auc=%.6f f1=%.6f",
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
) -> tuple[CatBoostClassifier, FoldEncoder]:
    """Train final CatBoost model on full train split."""
    x_train, x_valid, y_train, y_valid = stratified_train_validation_split(x_train_full, y_train_full)
    encoder = FoldEncoder().fit(x_train)
    x_train_encoded = encoder.transform(x_train)
    x_valid_encoded = encoder.transform(x_valid)
    model = CatBoostClassifier(**_catboost_params())
    model.fit(x_train_encoded, y_train, eval_set=(x_valid_encoded, y_valid), use_best_model=True)
    return model, encoder


def _save_model_bundle(model: CatBoostClassifier, encoder: FoldEncoder, threshold: float, trained_at: str) -> Path:
    """Save CatBoost model bundle without overwriting existing models."""
    ensure_directory(config.MODELS_SAVED_DIR)
    threshold_tag = f"{int(round(threshold * 100)):03d}"
    model_path = config.MODELS_SAVED_DIR / f"catboost_cv5_threshold{threshold_tag}_{trained_at}.pkl"
    payload = {
        "model": model,
        "encoder": encoder,
        "threshold": float(threshold),
        "version": "catboost_cv5",
        "trained_at": trained_at,
    }
    with model_path.open("wb") as output_file:
        pickle.dump(payload, output_file)
    return model_path


def _metrics_table(cv_summary: dict[str, float], holdout_metrics: dict[str, Any]) -> str:
    """Build markdown table for CV and holdout metrics."""
    rows = [
        ("Validation AUC (CV mean±std)", cv_summary["auc_roc_mean"], cv_summary["auc_roc_std"]),
        ("Holdout AUC", holdout_metrics["auc_roc"], None),
        ("Gini (CV mean±std)", cv_summary["gini_mean"], cv_summary["gini_std"]),
        ("KS (CV mean±std)", cv_summary["ks_statistic_mean"], cv_summary["ks_statistic_std"]),
        ("PR-AUC (CV mean±std)", cv_summary["pr_auc_mean"], cv_summary["pr_auc_std"]),
        ("Brier Score (CV mean±std)", cv_summary["brier_score_mean"], cv_summary["brier_score_std"]),
        ("Precision (CV mean±std)", cv_summary["precision_mean"], cv_summary["precision_std"]),
        ("Recall (CV mean±std)", cv_summary["recall_mean"], cv_summary["recall_std"]),
        ("F1-Score (CV mean±std)", cv_summary["f1_score_mean"], cv_summary["f1_score_std"]),
    ]
    lines = ["| Metric | CatBoost (threshold=0.30) |", "|---|---:|"]
    for name, val, std in rows:
        if std is None:
            lines.append(f"| {name} | {val:.6f} |")
        else:
            lines.append(f"| {name} | {val:.6f} +/- {std:.6f} |")
    return "\n".join(lines)


def _write_report(
    threshold: float,
    cv_summary: dict[str, float],
    fold_metrics: list[dict[str, Any]],
    holdout_metrics: dict[str, Any],
    model_path: Path,
) -> Path:
    """Write dedicated CatBoost report."""
    ensure_directory(config.REPORTS_DIR)
    report_path = config.REPORTS_DIR / "metrics_catboost_cv5.md"
    lines = [
        "# CatBoost CV5 Metrics",
        "",
        f"Threshold: `{threshold:.2f}`",
        "",
        _metrics_table(cv_summary, holdout_metrics),
        "",
        "## Holdout Confusion Matrix",
        f"- `{holdout_metrics['confusion_matrix']}`",
        "",
        "## Fold Details",
    ]
    for fold in fold_metrics:
        lines.append(
            f"- Fold {fold['fold']}: AUC={fold['auc_roc']:.6f}, PR-AUC={fold['pr_auc']:.6f}, "
            f"Precision={fold['precision']:.6f}, Recall={fold['recall']:.6f}, F1={fold['f1_score']:.6f}"
        )
    lines.extend(["", f"Saved model: `{model_path}`"])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def train_catboost_cv5(threshold: float = 0.30) -> dict[str, Any]:
    """Train CatBoost with 5-fold CV and evaluate at fixed threshold."""
    dataset_path = config.PROCESSED_DATA_DIR / config.OUTPUT_FILE_NAME
    if not dataset_path.exists():
        raise FileNotFoundError(f"Expected dataset not found: {dataset_path}")
    dataset = pd.read_parquet(dataset_path)
    x, y = _prepare_training_frame(dataset)

    x_train_full, x_holdout, y_train_full, y_holdout = stratified_holdout_split(x, y)
    fold_metrics, cv_summary = _cross_validate(x_train_full, y_train_full, threshold=threshold)

    final_model, final_encoder = _train_final_model(x_train_full, y_train_full)
    x_holdout_encoded = final_encoder.transform(x_holdout)
    holdout_score = final_model.predict_proba(x_holdout_encoded)[:, 1]
    holdout_metrics = _collect_metrics(y_holdout, holdout_score, threshold)

    trained_at = datetime.now().strftime(config.MODEL_DATE_FORMAT)
    model_path = _save_model_bundle(final_model, final_encoder, threshold, trained_at)
    report_path = _write_report(threshold, cv_summary, fold_metrics, holdout_metrics, model_path)

    return {
        "threshold": threshold,
        "model_path": str(model_path),
        "report_path": str(report_path),
        "cv_summary": cv_summary,
        "fold_metrics": fold_metrics,
        "holdout_metrics": holdout_metrics,
    }
