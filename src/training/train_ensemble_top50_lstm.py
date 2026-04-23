"""Ensemble: LightGBM Top-50 CV5 + LSTM OOF CV5 score blending."""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve

from sklearn.model_selection import train_test_split

from src.config import config
from src.data.cleaner import clean_dataframe
from src.data.loader import get_train_path, read_csv
from src.evaluation.metrics import calculate_classification_metrics_with_threshold
from src.features.interaction_features import add_interaction_features
from src.features.main_features import add_main_features
from src.utils.io import ensure_directory
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_lgbm_bundle() -> dict[str, Any]:
    candidates = sorted(config.MODELS_SAVED_DIR.glob("lgbm_top50_cv5_threshold*.pkl"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("lgbm_top50_cv5 model not found in models_saved/")
    path = candidates[-1]
    logger.info("Loading LGBM bundle: %s", path.name)
    with path.open("rb") as fh:
        return pickle.load(fh)


def _load_lstm_bundle_name() -> str:
    candidates = sorted(config.MODELS_SAVED_DIR.glob("lstm_oof_cv5_*.pt"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("lstm_oof_cv5 model not found in models_saved/")
    return candidates[-1].name


def _load_oof_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (customer_ids, oof_scores, y_true) for the full train split."""
    candidates = sorted(config.MODELS_SAVED_DIR.glob("lstm_oof_predictions_*.npz"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("lstm_oof_predictions not found in models_saved/")
    data = np.load(candidates[-1])
    return data["customer_ids"], data["oof_scores"], data["y_true"]


def _load_holdout_lstm_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (customer_ids, holdout_scores, y_true) for the holdout split."""
    candidates = sorted(config.MODELS_SAVED_DIR.glob("lstm_holdout_predictions_*.npz"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("lstm_holdout_predictions not found in models_saved/")
    data = np.load(candidates[-1])
    return data["customer_ids"], data["holdout_scores"], data["y_true"]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_lgbm(bundle: dict[str, Any], x_df: pd.DataFrame) -> np.ndarray:
    """Score with saved LGBM top-50 model.

    NaN values are intentionally kept — LightGBM handles them natively.
    Filling NaN with 0 would degrade predictions for customers with missing features.
    """
    enc = bundle["encoder"]
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    x_enc = enc.transform(x_df)  # no fillna — LGBM handles NaN natively
    present = [c for c in feature_cols if c in x_enc.columns]
    return model.predict_proba(x_enc[present])[:, 1].astype(np.float32)


# ---------------------------------------------------------------------------
# Blend weight & threshold search
# ---------------------------------------------------------------------------

def _find_f1_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5, 0.0
    f1_vals = 2.0 * (precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best = int(np.argmax(f1_vals))
    return float(thresholds[best]), float(f1_vals[best])


def _select_blend_weight_by_f1(
    y: np.ndarray,
    lgbm_scores: np.ndarray,
    lstm_scores: np.ndarray,
    threshold: float,
    step: float = 0.05,
) -> tuple[float, float]:
    """Grid-search LGBM weight (alpha) maximising F1 at given threshold."""
    best_w, best_f1 = 0.5, -1.0
    for w in np.arange(0.0, 1.0 + step, step):
        blended = float(w) * lgbm_scores + (1.0 - float(w)) * lstm_scores
        f1 = float(f1_score(y, (blended >= threshold).astype(int), zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_w = float(w)
    return best_w, best_f1


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(
    lgbm_bundle_name: str,
    lstm_bundle_name: str,
    lgbm_weight: float,
    blend_threshold: float,
    oof_metrics: dict[str, float],
    holdout_metrics: dict[str, Any],
    holdout_lgbm_metrics: dict[str, Any],
    holdout_lstm_metrics: dict[str, Any],
    trained_at: str,
    model_path: Path,
) -> Path:
    ensure_directory(config.REPORTS_DIR)
    report_path = config.REPORTS_DIR / f"ensemble_top50_lstm_report_{trained_at}.md"

    lstm_weight = 1.0 - lgbm_weight
    lines = [
        "# Ensemble Raporu — LightGBM Top-50 CV5 + LSTM OOF CV5",
        "",
        f"**Tarih:** {trained_at}  ",
        f"**Yöntem:** Score blending (ağırlıklı ortalama)  ",
        f"**LGBM modeli:** `{lgbm_bundle_name}`  ",
        f"**LSTM modeli:** `{lstm_bundle_name}`  ",
        f"**LGBM ağırlığı (α):** `{lgbm_weight:.2f}` | **LSTM ağırlığı (1-α):** `{lstm_weight:.2f}`  ",
        f"**Ensemble formülü:** `{lgbm_weight:.2f} × lgbm_score + {lstm_weight:.2f} × lstm_score`  ",
        f"**Threshold:** `{blend_threshold:.4f}` (OOF üzerinde F1 optimize edildi)  ",
        "",
        "## OOF Ağırlık Seçimi (train_full — leak-free)",
        "",
        "| Metrik | Değer |",
        "|---|---:|",
        f"| OOF Blend F1 @ {blend_threshold:.4f} | {oof_metrics['best_blend_f1']:.6f} |",
        f"| LGBM weight grid searched | 0.00 → 1.00 (step=0.05) |",
        "",
        "## Holdout Karşılaştırması",
        "",
        "| Model | AUC | Gini | KS | PR-AUC | F1 | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| LGBM Top-50 standalone | {holdout_lgbm_metrics['auc_roc']:.6f} | {holdout_lgbm_metrics['gini']:.6f} | {holdout_lgbm_metrics['ks_statistic']:.6f} | {holdout_lgbm_metrics['pr_auc']:.6f} | {holdout_lgbm_metrics['f1_at_threshold']:.6f} | {blend_threshold:.4f} |",
        f"| LSTM OOF CV5 standalone | {holdout_lstm_metrics['auc_roc']:.6f} | {holdout_lstm_metrics['gini']:.6f} | {holdout_lstm_metrics['ks_statistic']:.6f} | {holdout_lstm_metrics['pr_auc']:.6f} | {holdout_lstm_metrics['f1_at_threshold']:.6f} | {blend_threshold:.4f} |",
        f"| **Ensemble (α={lgbm_weight:.2f})** | **{holdout_metrics['auc_roc']:.6f}** | **{holdout_metrics['gini']:.6f}** | **{holdout_metrics['ks_statistic']:.6f}** | **{holdout_metrics['pr_auc']:.6f}** | **{holdout_metrics['f1_at_threshold']:.6f}** | **{blend_threshold:.4f}** |",
        "",
        "## Holdout Confusion Matrix",
        "",
        f"**Ensemble:** `{holdout_metrics['confusion_matrix']}`  ",
        f"**LGBM standalone:** `{holdout_lgbm_metrics['confusion_matrix']}`  ",
        f"**LSTM standalone:** `{holdout_lstm_metrics['confusion_matrix']}`  ",
        "",
        "## Kaydedilen Dosyalar",
        "",
        f"`{model_path}`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_ensemble_top50_lstm(threshold: float = 0.30) -> dict[str, Any]:
    """Blend LightGBM Top-50 + LSTM OOF CV5. Weight selected via OOF F1 grid search."""
    lgbm_bundle = _load_lgbm_bundle()
    lstm_name = _load_lstm_bundle_name()

    # Load LSTM pre-computed predictions
    oof_ids, oof_lstm_scores, oof_y = _load_oof_predictions()
    ho_ids, ho_lstm_scores, ho_y = _load_holdout_lstm_predictions()

    # Reproduce exact data split to get feature matrices for LGBM scoring
    logger.info("Loading and preparing data for LGBM scoring...")
    train_df = read_csv(get_train_path())
    dataset = add_interaction_features(add_main_features(clean_dataframe(train_df)))
    y_all = dataset[config.TARGET_COLUMN].astype(int)
    ids_all = dataset[config.KEY_COLUMN].astype(int)
    x_all = dataset.drop(columns=[config.TARGET_COLUMN, config.KEY_COLUMN], errors="ignore")

    x_train_full, x_holdout, y_train_full, y_holdout, ids_train_full, ids_holdout = train_test_split(
        x_all, y_all, ids_all,
        test_size=config.HOLDOUT_TEST_SIZE,
        stratify=y_all,
        random_state=config.RANDOM_SEED,
    )

    # Score LGBM on train_full and holdout
    logger.info("Scoring LGBM on train_full (%d rows)...", len(x_train_full))
    lgbm_train_scores = _score_lgbm(lgbm_bundle, x_train_full)
    logger.info("Scoring LGBM on holdout (%d rows)...", len(x_holdout))
    lgbm_holdout_scores = _score_lgbm(lgbm_bundle, x_holdout)

    # Both arrays share the same positional order (same train_test_split seed + data).
    # OOF scores are stored positionally (oof_scores[i] = row i of x_tf_reset = row i of x_train_full).
    # No customer_id merge needed — just verify lengths match.
    assert len(lgbm_train_scores) == len(oof_lstm_scores), (
        f"OOF length mismatch: lgbm={len(lgbm_train_scores)} lstm={len(oof_lstm_scores)}"
    )
    assert len(lgbm_holdout_scores) == len(ho_lstm_scores), (
        f"Holdout length mismatch: lgbm={len(lgbm_holdout_scores)} lstm={len(ho_lstm_scores)}"
    )
    logger.info("OOF positional alignment: %d rows | Holdout: %d rows",
                len(lgbm_train_scores), len(lgbm_holdout_scores))

    # Find F1-optimal threshold on blended OOF scores (initial pass with equal weights)
    init_blend = 0.5 * lgbm_train_scores + 0.5 * oof_lstm_scores
    blend_threshold, _ = _find_f1_optimal_threshold(oof_y, init_blend)
    blend_threshold = float(np.clip(blend_threshold, 0.05, 0.75))
    logger.info("Initial F1-optimal threshold (OOF equal weights): %.4f", blend_threshold)

    # Grid search LGBM weight at that threshold
    best_lgbm_weight, best_oof_f1 = _select_blend_weight_by_f1(
        oof_y, lgbm_train_scores, oof_lstm_scores, threshold=blend_threshold,
    )
    logger.info(
        "Best LGBM weight=%.2f → OOF F1=%.6f @ threshold=%.4f",
        best_lgbm_weight, best_oof_f1, blend_threshold,
    )

    # Refine threshold with the best weight
    best_blend_oof = best_lgbm_weight * lgbm_train_scores + (1.0 - best_lgbm_weight) * oof_lstm_scores
    blend_threshold, best_oof_f1 = _find_f1_optimal_threshold(oof_y, best_blend_oof)
    blend_threshold = float(np.clip(blend_threshold, 0.05, 0.75))
    logger.info("Refined threshold=%.4f | OOF F1=%.6f", blend_threshold, best_oof_f1)

    # Holdout: positional alignment (same split seed reproduces same order)
    y_holdout_arr = ho_y
    ensemble_holdout = best_lgbm_weight * lgbm_holdout_scores + (1.0 - best_lgbm_weight) * ho_lstm_scores

    # Compute holdout metrics for all three models
    ensemble_metrics = calculate_classification_metrics_with_threshold(
        y_holdout_arr, ensemble_holdout, blend_threshold
    )
    lgbm_standalone_metrics = calculate_classification_metrics_with_threshold(
        y_holdout_arr, lgbm_holdout_scores, blend_threshold
    )
    lstm_standalone_metrics = calculate_classification_metrics_with_threshold(
        y_holdout_arr, ho_lstm_scores, blend_threshold
    )

    logger.info(
        "Holdout — Ensemble AUC=%.6f F1=%.6f | LGBM AUC=%.6f F1=%.6f | LSTM AUC=%.6f F1=%.6f",
        ensemble_metrics["auc_roc"], ensemble_metrics["f1_at_threshold"],
        lgbm_standalone_metrics["auc_roc"], lgbm_standalone_metrics["f1_at_threshold"],
        lstm_standalone_metrics["auc_roc"], lstm_standalone_metrics["f1_at_threshold"],
    )
    # Sanity: verify holdout labels match between LGBM and LSTM splits
    y_lgbm_holdout = y_holdout.to_numpy()
    if not np.array_equal(y_lgbm_holdout, ho_y):
        logger.warning("Holdout y_true mismatch between split reproduction and LSTM npz — using LSTM npz labels")

    # Save ensemble bundle
    trained_at = datetime.now().strftime(config.MODEL_DATE_FORMAT)
    ensure_directory(config.MODELS_SAVED_DIR)
    bundle_path = config.MODELS_SAVED_DIR / f"ensemble_top50_lstm_{trained_at}.pkl"
    ensemble_bundle = {
        "lgbm_bundle_path": str(sorted(config.MODELS_SAVED_DIR.glob("lgbm_top50_cv5_threshold*.pkl"))[-1]),
        "lstm_bundle_path": str(sorted(config.MODELS_SAVED_DIR.glob("lstm_oof_cv5_*.pt"))[-1].name),
        "lgbm_weight": float(best_lgbm_weight),
        "lstm_weight": float(1.0 - best_lgbm_weight),
        "blend_threshold": float(blend_threshold),
        "trained_at": trained_at,
        "strategy": "score_blending_f1_oof",
        "metrics": {
            "oof_f1": float(best_oof_f1),
            "oof_threshold": float(blend_threshold),
            "holdout_ensemble_auc": float(ensemble_metrics["auc_roc"]),
            "holdout_ensemble_f1": float(ensemble_metrics["f1_at_threshold"]),
            "holdout_lgbm_auc": float(lgbm_standalone_metrics["auc_roc"]),
            "holdout_lgbm_f1": float(lgbm_standalone_metrics["f1_at_threshold"]),
            "holdout_lstm_auc": float(lstm_standalone_metrics["auc_roc"]),
            "holdout_lstm_f1": float(lstm_standalone_metrics["f1_at_threshold"]),
        },
    }
    with bundle_path.open("wb") as fh:
        pickle.dump(ensemble_bundle, fh)

    lgbm_name = sorted(config.MODELS_SAVED_DIR.glob("lgbm_top50_cv5_threshold*.pkl"))[-1].name
    report_path = _write_report(
        lgbm_bundle_name=lgbm_name,
        lstm_bundle_name=lstm_name,
        lgbm_weight=best_lgbm_weight,
        blend_threshold=blend_threshold,
        oof_metrics={"best_blend_f1": float(best_oof_f1)},
        holdout_metrics=ensemble_metrics,
        holdout_lgbm_metrics=lgbm_standalone_metrics,
        holdout_lstm_metrics=lstm_standalone_metrics,
        trained_at=trained_at,
        model_path=bundle_path,
    )
    logger.info("Ensemble report written: %s", report_path)

    # Save metadata JSON
    (config.MODELS_SAVED_DIR / f"ensemble_top50_lstm_metadata_{trained_at}.json").write_text(
        json.dumps(ensemble_bundle["metrics"] | {
            "lgbm_weight": best_lgbm_weight,
            "lstm_weight": 1.0 - best_lgbm_weight,
            "blend_threshold": blend_threshold,
            "trained_at": trained_at,
        }, indent=2),
        encoding="utf-8",
    )

    return {
        "lgbm_weight": float(best_lgbm_weight),
        "lstm_weight": float(1.0 - best_lgbm_weight),
        "blend_threshold": float(blend_threshold),
        "oof_f1": float(best_oof_f1),
        "ensemble_holdout_metrics": ensemble_metrics,
        "lgbm_standalone_metrics": lgbm_standalone_metrics,
        "lstm_standalone_metrics": lstm_standalone_metrics,
        "bundle_path": str(bundle_path),
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    train_ensemble_top50_lstm()
