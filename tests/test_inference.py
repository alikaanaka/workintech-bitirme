"""Inference layer unit tests — no real model files required."""

from __future__ import annotations

import pytest

from src.inference.risk_scorer import evaluate, proba_to_risk_score_pct, score_to_decision, score_to_risk_band
from tests.conftest import VALID_FEATURES, FakePredictor


# ---------------------------------------------------------------------------
# Risk scorer
# ---------------------------------------------------------------------------

def test_risk_band_low() -> None:
    assert score_to_risk_band(25) == "Low"


def test_risk_band_medium() -> None:
    assert score_to_risk_band(50) == "Medium"


def test_risk_band_high() -> None:
    assert score_to_risk_band(80) == "High"


def test_decision_approve() -> None:
    assert score_to_decision(25) == "ONAYLA"


def test_decision_review() -> None:
    assert score_to_decision(50) == "INCELE"


def test_decision_reject() -> None:
    assert score_to_decision(80) == "REDDET"


def test_proba_to_pct_clamped() -> None:
    assert proba_to_risk_score_pct(0.0) == 0
    assert proba_to_risk_score_pct(1.0) == 100
    assert proba_to_risk_score_pct(0.25) == 25


def test_evaluate_returns_tuple() -> None:
    band, pct, decision = evaluate(0.25)
    assert band == "Low"
    assert pct == 25
    assert decision == "ONAYLA"


# ---------------------------------------------------------------------------
# FakePredictor / predictor interface
# ---------------------------------------------------------------------------

def test_manual_mode_no_sequence() -> None:
    p = FakePredictor(lgbm_score=0.25)
    result = p.predict(VALID_FEATURES, prev_apps=None)
    assert result.proba_lgbm == pytest.approx(0.25)
    assert result.proba_lstm is None
    assert result.proba_ensemble is None
    assert "lgbm" in result.available_models


def test_available_models_lgbm_only() -> None:
    p = FakePredictor(lgbm_score=0.3, lstm_score=None)
    assert p.available_model_names == ["lgbm"]


def test_available_models_with_lstm() -> None:
    p = FakePredictor(lgbm_score=0.3, lstm_score=0.6)
    assert set(p.available_model_names) == {"lgbm", "lstm", "ensemble"}


def test_missing_models_when_all_loaded() -> None:
    p = FakePredictor(lgbm_score=0.3, lstm_score=0.6)
    assert p.missing_model_names == []


def test_missing_models_when_lstm_absent() -> None:
    p = FakePredictor(lgbm_score=0.3, lstm_score=None)
    missing = p.missing_model_names
    assert "lstm" in missing
    assert "ensemble" in missing


def test_sequence_mode_produces_ensemble() -> None:
    p = FakePredictor(lgbm_score=0.3, lstm_score=0.5)
    prev_apps = [{"AMT_CREDIT": 150000.0, "DAYS_DECISION": -365.0}]
    result = p.predict(VALID_FEATURES, prev_apps=prev_apps)
    assert result.proba_ensemble is not None
    expected = 0.55 * 0.3 + 0.45 * 0.5
    assert result.proba_ensemble == pytest.approx(expected, abs=1e-4)
