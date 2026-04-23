"""Tests for training pipeline utilities."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_classification_metrics
from src.evaluation.metrics import calculate_classification_metrics_with_threshold
from src.preprocessing.splitter import stratified_holdout_split, stratified_train_validation_split
from src.training.train_lgbm import _export_top_features, _prepare_training_frame


def test_stratified_split_preserves_target_ratio() -> None:
    """Split should be stratified for binary target."""
    x = pd.DataFrame({"a": np.arange(100)})
    y = pd.Series([0] * 80 + [1] * 20)
    x_train, x_holdout, y_train, y_holdout = stratified_holdout_split(x, y)

    assert len(x_train) == 80
    assert len(x_holdout) == 20
    assert abs(y_train.mean() - y.mean()) < 0.05
    assert abs(y_holdout.mean() - y.mean()) < 0.05


def test_train_validation_split_is_stratified() -> None:
    """Second split should keep train/validation class ratio close."""
    x = pd.DataFrame({"a": np.arange(120)})
    y = pd.Series([0] * 90 + [1] * 30)
    x_train, x_valid, y_train, y_valid = stratified_train_validation_split(x, y)

    assert len(x_train) == 96
    assert len(x_valid) == 24
    assert abs(y_train.mean() - y.mean()) < 0.05
    assert abs(y_valid.mean() - y.mean()) < 0.05


def test_prepare_training_frame_removes_key_column() -> None:
    """SK_ID_CURR should not be part of model features."""
    df = pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1], "F1": [10, 20]})
    x, y = _prepare_training_frame(df)
    assert "SK_ID_CURR" not in x.columns
    assert "TARGET" not in x.columns
    assert y.tolist() == [0, 1]


def test_top50_export_schema(tmp_path: Path) -> None:
    """Top feature export should follow expected JSON schema."""
    target_path = tmp_path / "top50_features.json"
    features = [f"f{i}" for i in range(60)]
    importances = np.linspace(60, 1, 60)
    exported = _export_top_features(features, importances, target_path)
    payload = json.loads(target_path.read_text(encoding="utf-8"))

    assert "features" in payload
    assert len(payload["features"]) == 50
    assert payload["features"][0]["importance_rank"] == 1
    assert payload["features"][0]["description"] is None
    assert len(exported) == 50


def test_metrics_function_returns_required_keys() -> None:
    """Metrics helper should expose required fields."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.8, 0.3, 0.7])
    metrics = calculate_classification_metrics(y_true, y_score)

    required = {
        "auc_roc",
        "gini",
        "ks_statistic",
        "pr_auc",
        "f1_optimal",
        "confusion_matrix",
        "brier_score",
        "calibration_summary",
    }
    assert required.issubset(metrics.keys())


def test_fixed_threshold_metrics_contains_expected_keys() -> None:
    """Fixed-threshold metrics should include threshold-specific keys."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.8, 0.3, 0.7])
    metrics = calculate_classification_metrics_with_threshold(y_true, y_score, threshold=0.22)

    assert "f1_at_threshold" in metrics
    assert "decision_threshold" in metrics
    assert metrics["decision_threshold"] == 0.22
