"""API endpoint tests — uses FakePredictor, no real model files required."""

from __future__ import annotations

import pytest

from tests.conftest import VALID_FEATURES


def test_health_ok(api_client) -> None:
    resp = api_client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "lgbm" in body["models_loaded"]


def test_features_top50_count(api_client) -> None:
    resp = api_client.get("/api/features/top50")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 50
    assert len(body["features"]) == 50
    first = body["features"][0]
    assert "rank" in first and "name" in first and "gain_norm" in first


def test_predict_valid_input_schema(api_client) -> None:
    resp = api_client.post("/api/predict", json=VALID_FEATURES)
    assert resp.status_code == 200
    body = resp.json()
    assert "model_version" in body
    assert "request_id" in body
    assert "proba_lgbm" in body
    assert "risk_band" in body
    assert "decision" in body
    assert "threshold_used" in body
    assert body["proba_lgbm"] is not None


def test_predict_manual_mode_no_sequence(api_client) -> None:
    """Without previous_applications, LSTM and ensemble scores must be null."""
    resp = api_client.post("/api/predict", json=VALID_FEATURES)
    assert resp.status_code == 200
    body = resp.json()
    assert body["proba_lstm"] is None
    assert body["proba_ensemble"] is None
    assert "lgbm" in body["available_models"]


def test_predict_invalid_ext_source_range(api_client) -> None:
    bad = {**VALID_FEATURES, "EXT_SOURCE_1": 1.5}
    resp = api_client.post("/api/predict", json=bad)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "field_errors" in detail
    assert any("EXT_SOURCE_1" in e["field"] for e in detail["field_errors"])


def test_predict_invalid_days_positive(api_client) -> None:
    bad = {**VALID_FEATURES, "DAYS_EMPLOYED": 100.0}
    resp = api_client.post("/api/predict", json=bad)
    assert resp.status_code == 422


def test_predict_cross_field_annuity_too_high(api_client) -> None:
    bad = {**VALID_FEATURES, "AMT_ANNUITY": 50000.0, "AMT_CREDIT": 100000.0}
    resp = api_client.post("/api/predict", json=bad)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "cross_field_errors" in detail


def test_predict_response_has_request_id_header(api_client) -> None:
    resp = api_client.post("/api/predict", json=VALID_FEATURES)
    assert "x-request-id" in resp.headers


def test_root(api_client) -> None:
    resp = api_client.get("/")
    assert resp.status_code == 200
    assert "docs" in resp.json()
