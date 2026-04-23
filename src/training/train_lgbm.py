"""Train LightGBM baseline and export artifacts."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

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
    drop_columns = [config.TARGET_COLUMN, config.KEY_COLUMN]
    x = dataset.drop(columns=[column for column in drop_columns if column in dataset.columns])
    return x, y


def _compute_scale_pos_weight(y_data: pd.Series) -> float:
    """Compute class weight as neg/pos."""
    pos_count = float((y_data == 1).sum())
    neg_count = float((y_data == 0).sum())
    return 1.0 if pos_count == 0 else neg_count / pos_count


def _export_top_features(feature_names: list[str], importances: np.ndarray, target_path: Path) -> list[dict]:
    """Export top-N feature importances with required schema."""
    ranking = sorted(zip(feature_names, importances), key=lambda pair: (-pair[1], pair[0]))
    top_ranking = ranking[: config.TOP_FEATURE_COUNT]
    total = float(np.sum([value for _, value in top_ranking]) or 1.0)
    features = []
    for rank, (name, value) in enumerate(top_ranking, start=1):
        feature_type = "numeric"
        if "__" in name or "_" in name and any(token in name for token in ("MISSING", "True", "False")):
            feature_type = "categorical_encoded"
        features.append(
            {
                "name": name,
                "type": feature_type,
                "importance_rank": rank,
                "importance_value": float(value / total),
                "display_name": None,
                "description": None,
                "default": None,
                "unit": None,
                "validation": None,
                "category_options": None,
            }
        )

    ensure_directory(target_path.parent)
    target_path.write_text(json.dumps({"features": features}, indent=2), encoding="utf-8")
    return features


def _write_metrics_report(
    validation_metrics: dict,
    holdout_metrics: dict,
    best_iteration: int,
    threshold: float,
    target_path: Path,
) -> None:
    """Write markdown metrics summary."""
    ensure_directory(target_path.parent)
    lines = [
        "# Metrics Comparison",
        "",
        "## LightGBM Baseline (Phase 2)",
        "",
        f"### Validation AUC: {validation_metrics['auc_roc']:.6f}",
        f"### Holdout Test AUC: {holdout_metrics['auc_roc']:.6f}",
        f"### Best Iteration: {best_iteration}",
        f"### Decision Threshold: {threshold:.2f}",
        "",
        "### Other Metrics (Holdout)",
        f"- Gini: {holdout_metrics['gini']:.6f}",
        f"- KS: {holdout_metrics['ks_statistic']:.6f}",
        f"- PR-AUC: {holdout_metrics['pr_auc']:.6f}",
        f"- F1@{threshold:.2f}: {holdout_metrics['f1_at_threshold']:.6f}",
        f"- Brier Score: {holdout_metrics['brier_score']:.6f}",
        "",
        "### Short Comment",
        "- Guclu yon: Holdout test ayrimi ve validation-tabanli early stopping ile sade baseline.",
        "- Risk: Tek split oldugu icin metrik varyansi CV kadar guvenilir olcumlenmez.",
        "",
        "## Placeholders",
        "- LSTM: TODO (Phase 3+)",
        "- Ensemble: TODO (Phase 3+)",
    ]
    target_path.write_text("\n".join(lines), encoding="utf-8")


def train_lgbm_baseline() -> dict:
    """Train LightGBM baseline from Phase 1 final dataset."""
    dataset_path = config.PROCESSED_DATA_DIR / config.OUTPUT_FILE_NAME
    if not dataset_path.exists():
        raise FileNotFoundError(f"Expected dataset not found: {dataset_path}")

    dataset = pd.read_parquet(dataset_path)
    x, y = _prepare_training_frame(dataset)
    max_rows_raw = os.getenv(config.TRAIN_MAX_ROWS_ENV)
    if max_rows_raw:
        max_rows = int(max_rows_raw)
        if 0 < max_rows < len(dataset):
            sampled = dataset.groupby(config.TARGET_COLUMN, group_keys=False).apply(
                lambda frame: frame.sample(
                    n=max(1, int(len(frame) * max_rows / len(dataset))),
                    random_state=config.RANDOM_SEED,
                )
            )
            sampled = sampled.sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)
            logger.warning("Using sampled dataset for training due to %s=%s", config.TRAIN_MAX_ROWS_ENV, max_rows)
            x, y = _prepare_training_frame(sampled)
    x_train_full, x_holdout, y_train_full, y_holdout = stratified_holdout_split(x, y)
    x_train, x_valid, y_train, y_valid = stratified_train_validation_split(x_train_full, y_train_full)
    scale_pos_weight = _compute_scale_pos_weight(y_train)
    logger.info("Train size=%s valid size=%s holdout size=%s", len(x_train), len(x_valid), len(x_holdout))
    logger.info("Computed scale_pos_weight=%.6f", scale_pos_weight)

    final_encoder = FoldEncoder()
    final_encoder.fit(x_train)
    x_train_encoded = final_encoder.transform(x_train)
    x_valid_encoded = final_encoder.transform(x_valid)
    x_holdout_encoded = final_encoder.transform(x_holdout)
    final_model = LGBMModel(params=config.LGBM_PARAMS, scale_pos_weight=scale_pos_weight)
    final_model.fit(x_train_encoded, y_train, x_valid_encoded, y_valid)

    validation_proba = final_model.predict_proba(x_valid_encoded)
    threshold = config.CLASSIFICATION_THRESHOLD
    validation_metrics = calculate_classification_metrics_with_threshold(y_valid.to_numpy(), validation_proba, threshold)
    holdout_proba = final_model.predict_proba(x_holdout_encoded)
    holdout_metrics = calculate_classification_metrics_with_threshold(y_holdout.to_numpy(), holdout_proba, threshold)

    ensure_directory(config.MODELS_SAVED_DIR)
    model_path = config.MODELS_SAVED_DIR / config.MODEL_FILE_NAME
    with model_path.open("wb") as model_file:
        pickle.dump({"model": final_model.model, "encoder": final_encoder}, model_file)

    feature_importances = final_model.model.booster_.feature_importance(importance_type="gain")
    top_features = _export_top_features(
        x_train_encoded.columns.tolist(),
        feature_importances,
        config.FEATURE_LISTS_DIR / config.TOP50_FEATURES_FILE_NAME,
    )

    _write_metrics_report(
        validation_metrics=validation_metrics,
        holdout_metrics=holdout_metrics,
        best_iteration=final_model.fit_result.best_iteration,
        threshold=threshold,
        target_path=config.REPORTS_DIR / config.METRICS_REPORT_FILE_NAME,
    )

    return {
        "validation_metrics": validation_metrics,
        "holdout_metrics": holdout_metrics,
        "best_iteration": final_model.fit_result.best_iteration,
        "decision_threshold": threshold,
        "model_path": str(model_path),
        "top_features_count": len(top_features),
    }
