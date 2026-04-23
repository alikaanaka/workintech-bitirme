"""API configuration — loads model_config.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "model_config.yaml"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load() -> dict:
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


_cfg = _load()

MODELS_SAVED_DIR: Path = _PROJECT_ROOT / "models_saved"
FEATURE_LISTS_DIR: Path = _PROJECT_ROOT / "data" / "artifacts" / "feature_lists"

LGBM_FILE: str = _cfg["models"]["lgbm"]["file"]
LSTM_FILE: str = _cfg["models"]["lstm"]["file"]
ENSEMBLE_FILE: str = _cfg["models"]["ensemble"]["file"]

LGBM_HOLDOUT_AUC: float = _cfg["models"]["lgbm"]["holdout_auc"]
LSTM_HOLDOUT_AUC: float = _cfg["models"]["lstm"]["holdout_auc"]
ENSEMBLE_HOLDOUT_AUC: float = _cfg["models"]["ensemble"]["holdout_auc"]

THRESHOLD_LGBM: float = _cfg["thresholds"]["lgbm"]
THRESHOLD_LSTM: float = _cfg["thresholds"]["lstm"]
THRESHOLD_ENSEMBLE: float = _cfg["thresholds"]["ensemble"]

ENSEMBLE_WEIGHT_LGBM: float = _cfg["ensemble_weights"]["lgbm"]
ENSEMBLE_WEIGHT_LSTM: float = _cfg["ensemble_weights"]["lstm"]

RISK_BAND_LOW_MAX: int = _cfg["risk_bands"]["low_max"]
RISK_BAND_MEDIUM_MAX: int = _cfg["risk_bands"]["medium_max"]

DECISION_APPROVE_MAX: int = _cfg["decisions"]["approve_max"]
DECISION_REVIEW_MAX: int = _cfg["decisions"]["review_max"]

API_VERSION: str = _cfg["api"]["version"]
CORS_ORIGINS: list[str] = _cfg["api"]["cors_origins"]

TOP50_FEATURE_LIST_FILE: str = "top50_features_lgbm_train_feature_cv5.json"
FEATURE_IMPORTANCE_LGBM_FILE: str = "feature_importance_lgbm_top50_cv5.json"
