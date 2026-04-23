"""Tests for ensemble fallback behavior."""

import numpy as np

from src.models.ensemble import blend_scores


def test_blend_scores_fallback_to_lgbm_when_sequence_missing() -> None:
    lgbm_score = np.array([0.1, 0.3, 0.8])
    result = blend_scores(lgbm_score, None, alpha=0.7)
    assert np.allclose(result, lgbm_score)


def test_blend_scores_combines_when_sequence_exists() -> None:
    lgbm_score = np.array([0.2, 0.4])
    lstm_score = np.array([0.6, 0.8])
    result = blend_scores(lgbm_score, lstm_score, alpha=0.5)
    assert np.allclose(result, np.array([0.4, 0.6]))
