"""Shared fixtures for API and inference tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Minimal valid feature dict (post-encoding, top-50 features)
# ---------------------------------------------------------------------------

VALID_FEATURES: dict[str, Any] = {
    "EXT_SOURCE_MEAN": 0.55,
    "EXT_SOURCE_1": 0.60,
    "EXT_SOURCE_2": 0.58,
    "EXT_SOURCE_3": 0.50,
    "EXT_SOURCE_PROD": 0.33,
    "EXT_SOURCE_STD": 0.05,
    "CREDIT_TERM": 36.0,
    "AMT_ANNUITY": 10000.0,
    "AMT_CREDIT": 200000.0,
    "AMT_GOODS_PRICE": 180000.0,
    "AMT_INCOME_TOTAL": 90000.0,
    "ANNUITY_INCOME_RATIO": 0.11,
    "CREDIT_INCOME_RATIO": 2.22,
    "INCOME_PER_PERSON": 45000.0,
    "DAYS_EMPLOYED": -1000.0,
    "DAYS_EMPLOYED_PERCENT": 0.15,
    "DAYS_BIRTH": -12000.0,
    "DAYS_ID_PUBLISH": -2000.0,
    "DAYS_LAST_PHONE_CHANGE": -500.0,
    "DAYS_REGISTRATION": -3000.0,
    "AGE_YEARS": 35.0,
    "OWN_CAR_AGE": 5.0,
    "REGION_POPULATION_RELATIVE": 0.025,
    "TOTALAREA_MODE": 0.08,
    "BASEMENTAREA_MODE": 0.05,
    "b_avg_utilization": 0.3,
    "b_avg_loan_duration": 24.0,
    "b_total_current_debt": 50000.0,
    "b_total_history_months": 60.0,
    "b_total_loan_count": 3.0,
    "b_active_loan_count": 1.0,
    "b_closed_loan_count": 2.0,
    "int_max_ins_days_late_ever": 0.0,
    "int_total_remaining_installments": 24.0,
    "int_avg_payment_performance": 0.98,
    "int_total_remaining_debt": 40000.0,
    "int_total_prev_loans_count": 3.0,
    "int_max_pos_dpd_ever": 0.0,
    "cc_total_avg_utilization_ratio": 0.25,
    "cc_total_credit_card_experience_months": 18.0,
    "cc_avg_repayment_performance": 0.95,
    "cc_total_transaction_count": 50.0,
    "cc_max_balance_ever": 30000.0,
    "cc_total_current_debt": 5000.0,
    "ORGANIZATION_TYPE_ENC": 5.0,
    "NAME_EDUCATION_TYPE_ENC": 2.0,
    "OCCUPATION_TYPE_ENC": 3.0,
    "CODE_GENDER_ENC": 0.0,
    "WEEKDAY_APPR_PROCESS_START_ENC": 2.0,
    "HOUR_APPR_PROCESS_START": 10.0,
}


class FakePredictor:
    """Predictor stub that returns fixed scores without loading real models."""

    def __init__(self, lgbm_score: float = 0.25, lstm_score: float | None = None) -> None:
        self._lgbm_score = lgbm_score
        self._lstm_score = lstm_score
        self._lgbm = MagicMock()
        self._ensemble = MagicMock(lgbm_weight=0.55, lstm_weight=0.45, threshold=0.6322)
        self._lstm = MagicMock() if lstm_score is not None else None

    @property
    def available_model_names(self) -> list[str]:
        names = ["lgbm"]
        if self._lstm is not None:
            names += ["lstm", "ensemble"]
        return names

    @property
    def missing_model_names(self) -> list[str]:
        available = set(self.available_model_names)
        return [m for m in ("lgbm", "lstm", "ensemble") if m not in available]

    def predict(self, features, prev_apps=None):
        from src.inference.predictor import InferenceResult
        from api import config as api_cfg

        proba_lstm = self._lstm_score if prev_apps and self._lstm is not None else None
        proba_ensemble = None
        if proba_lstm is not None:
            proba_ensemble = 0.55 * self._lgbm_score + 0.45 * proba_lstm

        threshold = api_cfg.THRESHOLD_ENSEMBLE if proba_ensemble is not None else api_cfg.THRESHOLD_LGBM

        return InferenceResult(
            proba_lgbm=self._lgbm_score,
            proba_lstm=proba_lstm,
            proba_ensemble=proba_ensemble,
            available_models=self.available_model_names,
            threshold_used=threshold,
            warnings=[],
        )

    def _score_lgbm(self, features: dict) -> float:
        return self._lgbm_score

    def load_all(self) -> None:
        pass


@pytest.fixture()
def fake_predictor() -> FakePredictor:
    return FakePredictor(lgbm_score=0.25)


@pytest.fixture()
def fake_predictor_high_risk() -> FakePredictor:
    return FakePredictor(lgbm_score=0.85)


@pytest.fixture()
def api_client(fake_predictor: FakePredictor) -> TestClient:
    """TestClient with the real FastAPI app but patched predictor (no real model loading)."""
    import src.inference.predictor as pred_module
    import api.routers.predict as predict_mod
    import api.routers.explain as explain_mod
    from api.main import app

    original_pred = pred_module.predictor

    # Replace the singleton BEFORE TestClient enters lifespan (lifespan reads _pred_module.predictor)
    pred_module.predictor = fake_predictor  # type: ignore[assignment]
    predict_mod.predictor = fake_predictor  # type: ignore[assignment]
    explain_mod.predictor = fake_predictor  # type: ignore[assignment]

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client

    pred_module.predictor = original_pred
    predict_mod.predictor = original_pred  # type: ignore[assignment]
    explain_mod.predictor = original_pred  # type: ignore[assignment]
