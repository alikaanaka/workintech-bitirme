"""FastAPI application — startup, CORS, logging, routes."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api import config as api_cfg
from api.routers import explain, features, predict
from api.schemas import HealthResponse
import src.inference.predictor as _pred_module

# ---------------------------------------------------------------------------
# Logging (JSON-style structured output)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "msg": %(message)s}',
)
logger = logging.getLogger("api")


# ---------------------------------------------------------------------------
# Lifespan — load models on startup, log on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info('"Loading model artifacts..."')
    try:
        _pred_module.predictor.load_all()
        logger.info('"Models loaded: %s"', _pred_module.predictor.available_model_names)
    except FileNotFoundError as exc:
        logger.error('"Critical artifact missing: %s"', exc)
        raise
    yield
    logger.info('"API shutting down."')


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Home Credit Risk API",
    description="Kredi temerrüt riski tahmini — LightGBM + LSTM ensemble",
    version=api_cfg.API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=api_cfg.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request-level middleware: attach request_id + log duration
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        '"method": "%s", "path": "%s", "status": %d, "ms": %.1f, "request_id": "%s"',
        request.method, request.url.path, response.status_code, elapsed, request_id,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(predict.router, prefix="/api")
app.include_router(explain.router, prefix="/api")
app.include_router(features.router, prefix="/api")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    p = _pred_module.predictor
    loaded = p.available_model_names
    missing = p.missing_model_names

    auc: dict[str, float] = {}
    if "lgbm" in loaded:
        auc["lgbm"] = api_cfg.LGBM_HOLDOUT_AUC
    if "lstm" in loaded:
        auc["lstm"] = api_cfg.LSTM_HOLDOUT_AUC
    if "ensemble" in loaded:
        auc["ensemble"] = api_cfg.ENSEMBLE_HOLDOUT_AUC

    thresholds: dict[str, float] = {}
    if "lgbm" in loaded:
        thresholds["lgbm"] = api_cfg.THRESHOLD_LGBM
    if "lstm" in loaded:
        thresholds["lstm"] = api_cfg.THRESHOLD_LSTM
    if "ensemble" in loaded:
        thresholds["ensemble"] = api_cfg.THRESHOLD_ENSEMBLE

    return HealthResponse(
        status="ok" if "lgbm" in loaded else "degraded",
        models_loaded=loaded,
        missing_models=missing,
        version=api_cfg.API_VERSION,
        auc_holdout=auc,
        thresholds=thresholds,
    )


@app.get("/")
def root():
    return {"message": "Home Credit Risk API", "docs": "/docs", "health": "/api/health"}
