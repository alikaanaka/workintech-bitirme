"""POST /api/explain — gain-importance-based feature contribution."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from api import config as api_cfg
from api.schemas import ExplainRequest, ExplainResponse, FeatureContribution
from src.inference.predictor import predictor

router = APIRouter()

_TOP_N = 10


def _load_importance() -> list[dict]:
    path = api_cfg.FEATURE_LISTS_DIR / api_cfg.FEATURE_IMPORTANCE_LGBM_FILE
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["features"]


def _direction(feature: str, value: float | None, gain_norm: float) -> str:
    """Heuristic: negative-days and utilization above 0.5 tend to increase risk."""
    if value is None:
        return "neutral"
    if feature.startswith("DAYS_") and value < -500:
        return "risk_reducing"
    if feature.startswith("EXT_SOURCE") and value < 0.5:
        return "risk_increasing"
    if "utilization" in feature and value > 0.5:
        return "risk_increasing"
    if "late" in feature and value > 0:
        return "risk_increasing"
    return "neutral"


@router.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest) -> ExplainResponse:
    if predictor._lgbm is None:
        raise HTTPException(status_code=503, detail="LGBM model not loaded.")

    importance_data = _load_importance()
    if not importance_data:
        raise HTTPException(status_code=503, detail="Feature importance file not found.")

    features_dict = req.model_dump(exclude_none=False)

    # Score LGBM
    try:
        proba_lgbm = predictor._score_lgbm(features_dict)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    top_contributions: list[FeatureContribution] = []
    for rank, entry in enumerate(importance_data[:_TOP_N], start=1):
        fname = entry["feature"]
        val = features_dict.get(fname)
        top_contributions.append(FeatureContribution(
            rank=rank,
            feature=fname,
            gain_norm=round(entry["gain_normalized"], 6),
            direction=_direction(fname, val, entry["gain_normalized"]),
        ))

    return ExplainResponse(
        explanation_type="importance_fallback",
        model_used="lgbm_top50_cv5",
        proba_lgbm=round(proba_lgbm, 6),
        top_features=top_contributions,
        warnings=["SHAP not available; directions are heuristic only."],
    )
