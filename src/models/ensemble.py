"""Ensemble helpers for LightGBM and LSTM scores."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score


def blend_scores(lgbm_score: np.ndarray, lstm_score: np.ndarray | None, alpha: float = 0.7) -> np.ndarray:
    """Blend scores; if no sequence score, fallback to LightGBM only."""
    if lstm_score is None:
        return lgbm_score.astype(float)
    return alpha * lgbm_score.astype(float) + (1.0 - alpha) * lstm_score.astype(float)


def select_lgbm_weight_by_auc(
    y_valid: np.ndarray,
    p_lgbm_valid: np.ndarray,
    p_lstm_valid: np.ndarray,
    step: float = 0.05,
) -> tuple[float, float]:
    """Grid-search LightGBM blend weight on validation AUC."""
    best_weight = 0.5
    best_auc = -np.inf
    grid = np.arange(0.0, 1.0 + step, step)
    for weight in grid:
        blended = blend_scores(p_lgbm_valid, p_lstm_valid, alpha=float(weight))
        auc = float(roc_auc_score(y_valid, blended))
        if auc > best_auc:
            best_auc = auc
            best_weight = float(weight)
    return best_weight, best_auc


def fit_isotonic_calibrator(y_valid: np.ndarray, p_valid: np.ndarray) -> IsotonicRegression:
    """Fit isotonic calibrator using validation predictions only."""
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(p_valid.astype(float), y_valid.astype(int))
    return calibrator
