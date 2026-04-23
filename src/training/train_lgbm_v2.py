"""Train and compare LightGBM v2 model with custom threshold."""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import config
from src.evaluation.metrics import calculate_classification_metrics_with_threshold
from src.models.lgbm_model import LGBMModel
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
    """Compute class weight as neg/pos."""
    pos_count = float((y_data == 1).sum())
    neg_count = float((y_data == 0).sum())
    return 1.0 if pos_count == 0 else neg_count / pos_count


def _load_dataset() -> pd.DataFrame:
    """Load processed Phase 1 dataset."""
    dataset_path = config.PROCESSED_DATA_DIR / config.OUTPUT_FILE_NAME
    if not dataset_path.exists():
        raise FileNotFoundError(f"Expected dataset not found: {dataset_path}")
    return pd.read_parquet(dataset_path)


def _load_v1_bundle() -> dict[str, Any]:
    """Load previously trained v1 model bundle."""
    v1_path = config.MODELS_SAVED_DIR / config.MODEL_FILE_NAME
    if not v1_path.exists():
        raise FileNotFoundError(f"v1 model file not found: {v1_path}")
    with v1_path.open("rb") as input_file:
        bundle = pickle.load(input_file)
    if "model" not in bundle or "encoder" not in bundle:
        raise ValueError("v1 model file format is invalid. Expected keys: model, encoder")
    return bundle


def _predict_with_bundle(bundle: dict[str, Any], x_data: pd.DataFrame) -> pd.Series:
    """Predict probability using stored model and encoder bundle."""
    encoder = bundle["encoder"]
    model = bundle["model"]
    transformed = encoder.transform(x_data)
    probabilities = model.predict_proba(transformed)[:, 1]
    return pd.Series(probabilities, index=x_data.index)


def _collect_metrics(y_true: pd.Series, y_score: pd.Series, threshold: float) -> dict[str, Any]:
    """Compute summary metrics at a fixed threshold."""
    metrics = calculate_classification_metrics_with_threshold(y_true.to_numpy(), y_score.to_numpy(), threshold)
    y_pred = (y_score.to_numpy() >= threshold).astype(int)
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0))
    return metrics


def _format_table(v1_validation: dict[str, Any], v2_validation: dict[str, Any], v1_holdout: dict[str, Any], v2_holdout: dict[str, Any]) -> str:
    """Create side-by-side markdown table for v1 and v2 metrics."""
    rows = [
        ("Validation AUC", v1_validation["auc_roc"], v2_validation["auc_roc"]),
        ("Holdout AUC", v1_holdout["auc_roc"], v2_holdout["auc_roc"]),
        ("Gini", v1_holdout["gini"], v2_holdout["gini"]),
        ("KS", v1_holdout["ks_statistic"], v2_holdout["ks_statistic"]),
        ("PR-AUC", v1_holdout["pr_auc"], v2_holdout["pr_auc"]),
        ("Brier Score", v1_holdout["brier_score"], v2_holdout["brier_score"]),
        ("Precision", v1_holdout["precision"], v2_holdout["precision"]),
        ("Recall", v1_holdout["recall"], v2_holdout["recall"]),
        ("F1-Score", v1_holdout["f1_score"], v2_holdout["f1_score"]),
    ]
    lines = [
        "| Metric | v1 @threshold | v2 @threshold |",
        "|---|---:|---:|",
    ]
    for name, v1_value, v2_value in rows:
        lines.append(f"| {name} | {v1_value:.6f} | {v2_value:.6f} |")
    return "\n".join(lines)


def _save_v2_bundle(model: Any, threshold: float, trained_at: str) -> Path:
    """Save v2 model bundle with threshold and metadata."""
    ensure_directory(config.MODELS_SAVED_DIR)
    threshold_tag = f"{int(round(threshold * 100)):03d}"
    model_path = config.MODELS_SAVED_DIR / f"lgbm_v2_threshold{threshold_tag}_{trained_at}.pkl"
    payload = {
        "model": model,
        "threshold": float(threshold),
        "version": config.MODEL_VERSION_V2,
        "trained_at": trained_at,
    }
    with model_path.open("wb") as output_file:
        pickle.dump(payload, output_file)
    return model_path


def _write_v2_report(
    threshold: float,
    comparison_table: str,
    v1_holdout_metrics: dict[str, Any],
    v2_holdout_metrics: dict[str, Any],
    model_path: Path,
) -> Path:
    """Persist v1-v2 comparison report to disk."""
    ensure_directory(config.REPORTS_DIR)
    report_path = config.REPORTS_DIR / config.METRICS_V2_REPORT_FILE_NAME
    lines = [
        "# Metrics Comparison V2",
        "",
        f"Threshold: `{threshold:.2f}`",
        "",
        comparison_table,
        "",
        "## Confusion Matrices (Holdout)",
        f"- v1: `{v1_holdout_metrics['confusion_matrix']}`",
        f"- v2: `{v2_holdout_metrics['confusion_matrix']}`",
        "",
        f"Saved model: `{model_path}`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def train_lgbm_v2(threshold: float) -> dict[str, Any]:
    """Train v2 model and compare against existing v1 model at same threshold."""
    dataset = _load_dataset()
    x, y = _prepare_training_frame(dataset)
    x_train_full, x_holdout, y_train_full, y_holdout = stratified_holdout_split(x, y)
    x_train, x_valid, y_train, y_valid = stratified_train_validation_split(x_train_full, y_train_full)

    scale_pos_weight = _compute_scale_pos_weight(y_train)
    logger.info("Train size=%s valid size=%s holdout size=%s", len(x_train), len(x_valid), len(x_holdout))
    logger.info("Computed scale_pos_weight=%.6f", scale_pos_weight)

    v2_encoder = FoldEncoder().fit(x_train)
    x_train_encoded = v2_encoder.transform(x_train)
    x_valid_encoded = v2_encoder.transform(x_valid)
    x_holdout_encoded = v2_encoder.transform(x_holdout)

    v2_model = LGBMModel(params=config.LGBM_PARAMS, scale_pos_weight=scale_pos_weight)
    v2_model.fit(x_train_encoded, y_train, x_valid_encoded, y_valid)
    v2_valid_proba = pd.Series(v2_model.predict_proba(x_valid_encoded), index=x_valid.index)
    v2_holdout_proba = pd.Series(v2_model.predict_proba(x_holdout_encoded), index=x_holdout.index)

    v1_bundle = _load_v1_bundle()
    v1_valid_proba = _predict_with_bundle(v1_bundle, x_valid)
    v1_holdout_proba = _predict_with_bundle(v1_bundle, x_holdout)

    v1_validation_metrics = _collect_metrics(y_valid, v1_valid_proba, threshold)
    v2_validation_metrics = _collect_metrics(y_valid, v2_valid_proba, threshold)
    v1_holdout_metrics = _collect_metrics(y_holdout, v1_holdout_proba, threshold)
    v2_holdout_metrics = _collect_metrics(y_holdout, v2_holdout_proba, threshold)

    trained_at = datetime.now().strftime(config.MODEL_DATE_FORMAT)
    model_path = _save_v2_bundle(v2_model.model, threshold, trained_at)

    metrics_table = _format_table(v1_validation_metrics, v2_validation_metrics, v1_holdout_metrics, v2_holdout_metrics)
    logger.info("Model comparison at threshold=%.2f\n%s", threshold, metrics_table)
    logger.info("v1 confusion matrix (holdout): %s", v1_holdout_metrics["confusion_matrix"])
    logger.info("v2 confusion matrix (holdout): %s", v2_holdout_metrics["confusion_matrix"])
    report_path = _write_v2_report(
        threshold=threshold,
        comparison_table=metrics_table,
        v1_holdout_metrics=v1_holdout_metrics,
        v2_holdout_metrics=v2_holdout_metrics,
        model_path=model_path,
    )

    return {
        "threshold": threshold,
        "model_path": str(model_path),
        "report_path": str(report_path),
        "v1_validation_metrics": v1_validation_metrics,
        "v2_validation_metrics": v2_validation_metrics,
        "v1_holdout_metrics": v1_holdout_metrics,
        "v2_holdout_metrics": v2_holdout_metrics,
        "comparison_table": metrics_table,
    }
