"""Evaluation metrics helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.evaluation.calibration import expected_calibration_error


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute KS statistic from ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return threshold that maximizes F1 and the F1 value."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5, 0.0
    f1_values = 2 * (precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def _calibration_summary(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> dict:
    """Provide lightweight calibration summary."""
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.digitize(y_score, bin_edges, right=True)
    observed = []
    predicted = []
    for bin_id in range(1, bins + 1):
        mask = indices == bin_id
        if np.any(mask):
            observed.append(float(np.mean(y_true[mask])))
            predicted.append(float(np.mean(y_score[mask])))
    calibration_gap = float(np.mean(np.abs(np.array(observed) - np.array(predicted)))) if observed else 0.0
    return {"mean_abs_calibration_gap": calibration_gap, "non_empty_bins": len(observed)}


def calculate_classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Calculate binary classification metrics for model scores."""
    auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    gini = float(2 * auc - 1)
    ks_value = _ks_statistic(y_true, y_score)
    best_threshold, best_f1 = _best_f1_threshold(y_true, y_score)
    y_pred = (y_score >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    brier = float(brier_score_loss(y_true, y_score))
    calibration = _calibration_summary(y_true, y_score)
    return {
        "auc_roc": auc,
        "gini": gini,
        "ks_statistic": ks_value,
        "pr_auc": pr_auc,
        "f1_optimal": best_f1,
        "optimal_threshold": best_threshold,
        "confusion_matrix": cm.tolist(),
        "brier_score": brier,
        "calibration_summary": calibration,
        "ece": expected_calibration_error(y_true, y_score),
    }


def calculate_classification_metrics_with_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict:
    """Calculate metrics using a fixed decision threshold."""
    auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    gini = float(2 * auc - 1)
    ks_value = _ks_statistic(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    f1_fixed = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_score))
    calibration = _calibration_summary(y_true, y_score)
    return {
        "auc_roc": auc,
        "gini": gini,
        "ks_statistic": ks_value,
        "pr_auc": pr_auc,
        "f1_at_threshold": f1_fixed,
        "decision_threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "brier_score": brier,
        "calibration_summary": calibration,
        "ece": expected_calibration_error(y_true, y_score),
    }
