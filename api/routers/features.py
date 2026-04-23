"""GET /api/features/top50 — return the top-50 LGBM feature list."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from api import config as api_cfg
from api.schemas import FeatureEntry, FeaturesResponse

router = APIRouter()


@router.get("/features/top50", response_model=FeaturesResponse)
def get_top50_features() -> FeaturesResponse:
    path = api_cfg.FEATURE_LISTS_DIR / api_cfg.FEATURE_IMPORTANCE_LGBM_FILE
    if not path.exists():
        raise HTTPException(status_code=503, detail="Feature importance file not found.")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    entries = [
        FeatureEntry(
            rank=f["rank"],
            name=f["feature"],
            gain_norm=round(f["gain_normalized"], 6),
            split_norm=round(f["split_normalized"], 6),
        )
        for f in data["features"]
    ]
    return FeaturesResponse(count=len(entries), features=entries)
