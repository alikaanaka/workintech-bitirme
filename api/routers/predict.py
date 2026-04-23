"""POST /api/predict — main prediction endpoint."""

from __future__ import annotations

import uuid

import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api import config as api_cfg
from api.schemas import PredictRequest, PredictResponse
from api.validators import run_all_validations
from src.inference.predictor import predictor
from src.inference.risk_scorer import evaluate

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    request_id = str(uuid.uuid4())

    # --- validation ---
    field_errors, cross_errors = run_all_validations(req)
    if field_errors or cross_errors:
        detail = {
            "field_errors": [e.model_dump() for e in field_errors],
            "cross_field_errors": [e.model_dump() for e in cross_errors],
        }
        raise HTTPException(status_code=422, detail=detail)

    if predictor._lgbm is None:
        raise HTTPException(status_code=503, detail="LGBM model not loaded.")

    # --- build feature dict (top-50 values) ---
    features_dict = req.model_dump(exclude={"previous_applications"}, exclude_none=False)

    prev_apps: list[dict] | None = None
    if req.previous_applications:
        prev_apps = [app.model_dump() for app in req.previous_applications]

    # --- inference ---
    t0 = time.perf_counter()
    try:
        result = predictor.predict(features_dict, prev_apps)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # --- risk evaluation ---
    # Use ensemble proba if available, else LGBM
    eval_proba = result.proba_ensemble if result.proba_ensemble is not None else result.proba_lgbm
    risk_band, risk_score_pct, decision = evaluate(eval_proba)

    model_version = (
        "ensemble_top50_lstm_20260422" if result.proba_ensemble is not None else "lgbm_top50_cv5_20260421"
    )

    return PredictResponse(
        model_version=model_version,
        request_id=request_id,
        inference_time_ms=round(elapsed_ms, 2),
        proba_lgbm=round(result.proba_lgbm, 6),
        proba_lstm=round(result.proba_lstm, 6) if result.proba_lstm is not None else None,
        proba_ensemble=round(result.proba_ensemble, 6) if result.proba_ensemble is not None else None,
        available_models=result.available_models,
        risk_band=risk_band,
        risk_score_pct=risk_score_pct,
        decision=decision,
        threshold_used=result.threshold_used,
        warnings=result.warnings,
    )
