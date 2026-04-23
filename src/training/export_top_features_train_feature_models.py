"""Export top-50 feature lists for train_feature-only model bundles."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import config
from src.data.cleaner import clean_dataframe
from src.data.loader import get_train_path, read_csv
from src.features.interaction_features import add_interaction_features
from src.features.main_features import add_main_features
from src.utils.io import ensure_directory


def _load_train_feature_matrix() -> pd.DataFrame:
    """Load and prepare train_feature matrix to recover encoded column names."""
    dataset = read_csv(get_train_path())
    dataset_clean = clean_dataframe(dataset)
    dataset_featured = add_interaction_features(add_main_features(dataset_clean))
    return dataset_featured.drop(columns=[config.TARGET_COLUMN, config.KEY_COLUMN], errors="ignore")


def _latest_model_path(pattern: str) -> Path:
    """Return latest model path by modified time for given pattern."""
    candidates = list(config.MODELS_SAVED_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No model files found for pattern: {pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _get_importances(model: Any) -> np.ndarray:
    """Extract feature importance vector from supported model types."""
    if hasattr(model, "get_feature_importance"):
        return np.asarray(model.get_feature_importance(), dtype=float)
    if hasattr(model, "booster_"):
        return np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    raise ValueError(f"Unsupported model type for feature importance: {type(model)}")


def _feature_type(name: str) -> str:
    """Infer lightweight feature type for exported schema."""
    if "__" in name or ("_" in name and any(token in name for token in ("MISSING", "True", "False", "_ENC"))):
        return "categorical_encoded"
    return "numeric"


def _export_top50(feature_names: list[str], importances: np.ndarray, target_path: Path) -> None:
    """Write normalized top-50 feature list to JSON."""
    ranking = sorted(zip(feature_names, importances), key=lambda pair: (-pair[1], pair[0]))[:50]
    total = float(np.sum([value for _, value in ranking]) or 1.0)
    payload = {
        "features": [
            {
                "name": name,
                "type": _feature_type(name),
                "importance_rank": idx + 1,
                "importance_value": float(value / total),
                "display_name": None,
                "description": None,
                "default": None,
                "unit": None,
                "validation": None,
                "category_options": None,
            }
            for idx, (name, value) in enumerate(ranking)
        ]
    }
    ensure_directory(target_path.parent)
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _process_model(pattern: str, output_file_name: str, x_features: pd.DataFrame) -> Path:
    """Load latest bundle, extract encoded features and save top50 list."""
    model_path = _latest_model_path(pattern)
    with model_path.open("rb") as input_file:
        bundle = pickle.load(input_file)
    model = bundle["model"]
    encoder = bundle["encoder"]
    encoded_columns = encoder.transform(x_features.head(500)).columns.tolist()
    importances = _get_importances(model)
    if len(encoded_columns) != len(importances):
        raise ValueError(
            f"Feature count mismatch for {model_path.name}: columns={len(encoded_columns)} importances={len(importances)}"
        )
    output_path = config.FEATURE_LISTS_DIR / output_file_name
    _export_top50(encoded_columns, importances, output_path)
    return output_path


def export_top_features_train_feature_models() -> dict[str, str]:
    """Export top-50 feature files for CatBoost/XGBoost/LightGBM train_feature models."""
    x_features = _load_train_feature_matrix()
    catboost_out = _process_model(
        pattern="catboost_train_feature_cv5_threshold*.pkl",
        output_file_name="top50_features_catboost_train_feature_cv5.json",
        x_features=x_features,
    )
    xgboost_out = _process_model(
        pattern="xgboost_train_feature_cv5_threshold*.pkl",
        output_file_name="top50_features_xgboost_train_feature_cv5.json",
        x_features=x_features,
    )
    lgbm_out = _process_model(
        pattern="lgbm_train_feature_cv5_threshold*.pkl",
        output_file_name="top50_features_lgbm_train_feature_cv5.json",
        x_features=x_features,
    )
    return {
        "catboost": str(catboost_out),
        "xgboost": str(xgboost_out),
        "lgbm": str(lgbm_out),
    }

