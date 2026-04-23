"""Train LightGBM with scale_pos_weight and top-50 features, 5-fold CV."""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from src.config import config
from src.data.cleaner import clean_dataframe
from src.data.loader import get_train_path, read_csv
from src.evaluation.metrics import calculate_classification_metrics_with_threshold
from src.features.interaction_features import add_interaction_features
from src.features.main_features import add_main_features
from src.preprocessing.encoder import FoldEncoder
from src.preprocessing.splitter import stratified_holdout_split, stratified_train_validation_split
from src.utils.io import ensure_directory
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TOP50_FEATURE_PATH = config.FEATURE_LISTS_DIR / "top50_features_lgbm_train_feature_cv5.json"
_MODEL_TAG = "lgbm_top50_cv5"
_THRESHOLD_GRID = np.arange(0.10, 0.91, 0.05)


def _load_top50_names() -> list[str]:
    """Return feature names from top-50 JSON artifact."""
    with _TOP50_FEATURE_PATH.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return [entry["name"] for entry in payload["features"]]


def _filter_top50(encoded_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Keep only columns whose names are in feature_names (order-preserving)."""
    present = [col for col in feature_names if col in encoded_df.columns]
    missing = [col for col in feature_names if col not in encoded_df.columns]
    if missing:
        logger.warning("Top-50 columns absent after encoding: %s", missing)
    return encoded_df[present]


def _prepare_training_frame(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and target vector."""
    y = dataset[config.TARGET_COLUMN].astype(int)
    x = dataset.drop(columns=[config.TARGET_COLUMN, config.KEY_COLUMN], errors="ignore")
    return x, y


def _compute_scale_pos_weight(y_series: pd.Series) -> float:
    pos = float((y_series == 1).sum())
    neg = float((y_series == 0).sum())
    return 1.0 if pos == 0 else neg / pos


def _lgbm_params(scale_pos_weight: float) -> dict[str, Any]:
    params = dict(config.LGBM_PARAMS)
    params["scale_pos_weight"] = float(scale_pos_weight)
    return params


def _collect_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    base = calculate_classification_metrics_with_threshold(y_true.to_numpy(), y_score, threshold)
    y_pred = (y_score >= threshold).astype(int)
    base["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    base["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    base["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0))
    return base


def _cross_validate(
    x_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    top50: list[str],
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """5-fold stratified CV using only top-50 features and scale_pos_weight."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    fold_metrics: list[dict[str, Any]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(x_train_full, y_train_full), start=1):
        x_tr = x_train_full.iloc[train_idx]
        y_tr = y_train_full.iloc[train_idx]
        x_va = x_train_full.iloc[valid_idx]
        y_va = y_train_full.iloc[valid_idx]

        enc = FoldEncoder().fit(x_tr)
        x_tr_enc = _filter_top50(enc.transform(x_tr), top50)
        x_va_enc = _filter_top50(enc.transform(x_va), top50)

        spw = _compute_scale_pos_weight(y_tr)
        model = lgb.LGBMClassifier(**_lgbm_params(spw), n_estimators=config.LGBM_NUM_BOOST_ROUND)
        model.fit(
            x_tr_enc,
            y_tr,
            eval_set=[(x_va_enc, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(config.LGBM_EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        val_score = model.predict_proba(x_va_enc)[:, 1]
        metrics = _collect_metrics(y_va, val_score, threshold)
        metrics["fold"] = fold_idx
        metrics["scale_pos_weight"] = float(spw)
        fold_metrics.append(metrics)
        logger.info(
            "top50 CV fold=%s auc=%.6f pr_auc=%.6f f1=%.6f spw=%.2f",
            fold_idx,
            metrics["auc_roc"],
            metrics["pr_auc"],
            metrics["f1_score"],
            spw,
        )

    summary_keys = ["auc_roc", "gini", "ks_statistic", "pr_auc", "brier_score", "precision", "recall", "f1_score"]
    summary: dict[str, float] = {}
    for key in summary_keys:
        vals = [f[key] for f in fold_metrics]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
    return fold_metrics, summary


def _train_final_model(
    x_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    top50: list[str],
) -> tuple[lgb.LGBMClassifier, FoldEncoder, float, list[str]]:
    """Train final model on the full train split."""
    x_tr, x_va, y_tr, y_va = stratified_train_validation_split(x_train_full, y_train_full)
    enc = FoldEncoder().fit(x_tr)
    x_tr_enc = _filter_top50(enc.transform(x_tr), top50)
    x_va_enc = _filter_top50(enc.transform(x_va), top50)
    spw = _compute_scale_pos_weight(y_tr)
    model = lgb.LGBMClassifier(**_lgbm_params(spw), n_estimators=config.LGBM_NUM_BOOST_ROUND)
    model.fit(
        x_tr_enc,
        y_tr,
        eval_set=[(x_va_enc, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(config.LGBM_EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model, enc, float(spw), list(x_tr_enc.columns)


def _save_model_bundle(
    model: lgb.LGBMClassifier,
    encoder: FoldEncoder,
    feature_columns: list[str],
    top50_names: list[str],
    threshold: float,
    trained_at: str,
    scale_pos_weight: float,
) -> Path:
    ensure_directory(config.MODELS_SAVED_DIR)
    tag = f"{int(round(threshold * 100)):03d}"
    path = config.MODELS_SAVED_DIR / f"{_MODEL_TAG}_threshold{tag}_{trained_at}.pkl"
    payload = {
        "model": model,
        "encoder": encoder,
        "feature_columns": feature_columns,
        "top50_names": top50_names,
        "threshold": float(threshold),
        "version": _MODEL_TAG,
        "trained_at": trained_at,
        "scale_pos_weight": float(scale_pos_weight),
    }
    with path.open("wb") as fh:
        pickle.dump(payload, fh)
    return path


def _threshold_table(y_true: pd.Series, y_score: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for thr in _THRESHOLD_GRID:
        y_pred = (y_score >= thr).astype(int)
        rows.append({
            "threshold": float(thr),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        })
    return rows


def _save_precision_recall_chart(
    y_true: pd.Series,
    y_score: np.ndarray,
    target_path: Path,
    threshold_rows: list[dict[str, float]],
    fixed_threshold: float,
) -> None:
    """Save PR curve with annotated threshold markers."""
    ensure_directory(target_path.parent)
    precision_curve, recall_curve, curve_thresholds = precision_recall_curve(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: PR curve with threshold markers
    ax = axes[0]
    ax.plot(recall_curve, precision_curve, color="steelblue", linewidth=2, label="PR Curve")
    baseline = float((y_true == 1).mean())
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label=f"Baseline ({baseline:.3f})")

    marker_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    cmap = plt.cm.get_cmap("RdYlGn", len(marker_thresholds))
    for i, thr in enumerate(marker_thresholds):
        idx = np.searchsorted(curve_thresholds, thr, side="left")
        idx = min(idx, len(recall_curve) - 2)
        r, p = recall_curve[idx], precision_curve[idx]
        color = "darkred" if thr == fixed_threshold else cmap(i)
        ax.scatter(r, p, s=80, zorder=5, color=color)
        offset = (0.01, 0.01) if i % 2 == 0 else (0.01, -0.04)
        ax.annotate(
            f"t={thr:.2f}\nP={p:.2f} R={r:.2f}",
            xy=(r, p),
            xytext=(r + offset[0], p + offset[1]),
            fontsize=7,
            color=color,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve\n(LightGBM Top-50 CV5)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Right: F1 / Precision / Recall by threshold
    ax2 = axes[1]
    thresholds_x = [r["threshold"] for r in threshold_rows]
    precisions = [r["precision"] for r in threshold_rows]
    recalls = [r["recall"] for r in threshold_rows]
    f1s = [r["f1"] for r in threshold_rows]

    ax2.plot(thresholds_x, precisions, color="steelblue", marker="o", markersize=4, label="Precision")
    ax2.plot(thresholds_x, recalls, color="darkorange", marker="s", markersize=4, label="Recall")
    ax2.plot(thresholds_x, f1s, color="forestgreen", marker="^", markersize=5, linewidth=2, label="F1")
    ax2.axvline(fixed_threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold={fixed_threshold:.2f}")

    best_f1_idx = int(np.argmax(f1s))
    best_thr = thresholds_x[best_f1_idx]
    ax2.axvline(best_thr, color="purple", linestyle=":", linewidth=1.5, label=f"Best F1 t={best_thr:.2f}")

    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Precision / Recall / F1 by Threshold")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("LightGBM Top-50 Features — Threshold Analysis (Holdout)", fontsize=12)
    plt.tight_layout()
    fig.savefig(target_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    threshold: float,
    cv_summary: dict[str, float],
    fold_metrics: list[dict[str, Any]],
    holdout_metrics: dict[str, Any],
    model_path: Path,
    scale_pos_weight: float,
    pr_chart_path: Path,
    threshold_rows: list[dict[str, float]],
    top50_names: list[str],
    trained_at: str,
) -> Path:
    ensure_directory(config.REPORTS_DIR)
    report_path = config.REPORTS_DIR / f"metrics_{_MODEL_TAG}_{trained_at}.md"

    best_f1_row = max(threshold_rows, key=lambda r: r["f1"])

    lines = [
        f"# LightGBM Top-50 Features CV5 — Metrics Report",
        f"",
        f"**Trained at:** {trained_at}  ",
        f"**Fixed threshold:** `{threshold:.2f}`  ",
        f"**scale_pos_weight (final split):** `{scale_pos_weight:.4f}`  ",
        f"**Features:** {len(top50_names)} (top-50 by gain from prior CV5 model)  ",
        f"",
        f"## Cross-Validation Summary (5-Fold)",
        f"",
        f"| Metric | Mean | Std |",
        f"|---|---:|---:|",
        f"| AUC-ROC | {cv_summary['auc_roc_mean']:.6f} | {cv_summary['auc_roc_std']:.6f} |",
        f"| Gini | {cv_summary['gini_mean']:.6f} | {cv_summary['gini_std']:.6f} |",
        f"| KS | {cv_summary['ks_statistic_mean']:.6f} | {cv_summary['ks_statistic_std']:.6f} |",
        f"| PR-AUC | {cv_summary['pr_auc_mean']:.6f} | {cv_summary['pr_auc_std']:.6f} |",
        f"| Brier Score | {cv_summary['brier_score_mean']:.6f} | {cv_summary['brier_score_std']:.6f} |",
        f"| Precision | {cv_summary['precision_mean']:.6f} | {cv_summary['precision_std']:.6f} |",
        f"| Recall | {cv_summary['recall_mean']:.6f} | {cv_summary['recall_std']:.6f} |",
        f"| F1-Score | {cv_summary['f1_score_mean']:.6f} | {cv_summary['f1_score_std']:.6f} |",
        f"",
        f"## Holdout Metrics (threshold={threshold:.2f})",
        f"",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| AUC-ROC | {holdout_metrics['auc_roc']:.6f} |",
        f"| Gini | {holdout_metrics['gini']:.6f} |",
        f"| KS | {holdout_metrics['ks_statistic']:.6f} |",
        f"| PR-AUC | {holdout_metrics['pr_auc']:.6f} |",
        f"| Precision | {holdout_metrics['precision']:.6f} |",
        f"| Recall | {holdout_metrics['recall']:.6f} |",
        f"| F1-Score | {holdout_metrics['f1_score']:.6f} |",
        f"| Brier Score | {holdout_metrics['brier_score']:.6f} |",
        f"",
        f"**Confusion Matrix (holdout):** `{holdout_metrics['confusion_matrix']}`",
        f"",
        f"## Best F1 Threshold (Holdout)",
        f"",
        f"| Threshold | Precision | Recall | F1 |",
        f"|---:|---:|---:|---:|",
        f"| **{best_f1_row['threshold']:.2f}** | {best_f1_row['precision']:.6f} | {best_f1_row['recall']:.6f} | **{best_f1_row['f1']:.6f}** |",
        f"",
        f"## Precision-Recall Chart",
        f"",
        f"![PR Curve]({pr_chart_path.name})",
        f"",
        f"Chart saved: `{pr_chart_path}`",
        f"",
        f"## Precision / Recall / F1 by Threshold (Holdout)",
        f"",
        f"| Threshold | Precision | Recall | F1 |",
        f"|---:|---:|---:|---:|",
    ]
    for row in threshold_rows:
        marker = " ◀" if abs(row["threshold"] - threshold) < 0.001 else ""
        lines.append(
            f"| {row['threshold']:.2f} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f}{marker} |"
        )

    lines.extend([
        f"",
        f"## Fold Details",
        f"",
    ])
    for fold in fold_metrics:
        lines.append(
            f"- **Fold {fold['fold']}**: AUC={fold['auc_roc']:.6f}, PR-AUC={fold['pr_auc']:.6f}, "
            f"Precision={fold['precision']:.6f}, Recall={fold['recall']:.6f}, "
            f"F1={fold['f1_score']:.6f}, scale_pos_weight={fold['scale_pos_weight']:.2f}"
        )

    lines.extend([
        f"",
        f"## Top-50 Feature Names Used",
        f"",
        ", ".join(f"`{n}`" for n in top50_names),
        f"",
        f"## Saved Model",
        f"",
        f"`{model_path}`",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def train_lgbm_top50_cv5(threshold: float = 0.30) -> dict[str, Any]:
    """Train LightGBM with top-50 features, scale_pos_weight, and 5-fold CV."""
    top50_names = _load_top50_names()
    logger.info("Loaded %d top features from %s", len(top50_names), _TOP50_FEATURE_PATH)

    train_path = get_train_path()
    dataset = read_csv(train_path)
    dataset_clean = clean_dataframe(dataset)
    dataset_featured = add_interaction_features(add_main_features(dataset_clean))

    x, y = _prepare_training_frame(dataset_featured)
    x_train_full, x_holdout, y_train_full, y_holdout = stratified_holdout_split(x, y)

    fold_metrics, cv_summary = _cross_validate(x_train_full, y_train_full, top50_names, threshold)

    final_model, final_encoder, final_spw, feature_columns = _train_final_model(x_train_full, y_train_full, top50_names)

    x_holdout_enc = _filter_top50(final_encoder.transform(x_holdout), top50_names)
    holdout_score = final_model.predict_proba(x_holdout_enc)[:, 1]
    holdout_metrics = _collect_metrics(y_holdout, holdout_score, threshold)

    logger.info(
        "Holdout: AUC=%.6f  PR-AUC=%.6f  F1=%.6f  Precision=%.6f  Recall=%.6f",
        holdout_metrics["auc_roc"],
        holdout_metrics["pr_auc"],
        holdout_metrics["f1_score"],
        holdout_metrics["precision"],
        holdout_metrics["recall"],
    )

    trained_at = datetime.now().strftime(config.MODEL_DATE_FORMAT)
    model_path = _save_model_bundle(
        final_model, final_encoder, feature_columns, top50_names, threshold, trained_at, final_spw
    )

    pr_chart_path = config.REPORTS_DIR / f"{_MODEL_TAG}_pr_threshold_chart_{trained_at}.png"
    threshold_rows = _threshold_table(y_holdout, holdout_score)
    _save_precision_recall_chart(y_holdout, holdout_score, pr_chart_path, threshold_rows, threshold)

    report_path = _write_report(
        threshold, cv_summary, fold_metrics, holdout_metrics,
        model_path, final_spw, pr_chart_path, threshold_rows, top50_names, trained_at,
    )
    logger.info("Report written: %s", report_path)

    return {
        "threshold": threshold,
        "model_path": str(model_path),
        "report_path": str(report_path),
        "pr_chart_path": str(pr_chart_path),
        "cv_summary": cv_summary,
        "fold_metrics": fold_metrics,
        "holdout_metrics": holdout_metrics,
        "scale_pos_weight": final_spw,
        "top50_names": top50_names,
        "threshold_rows": threshold_rows,
    }


if __name__ == "__main__":
    train_lgbm_top50_cv5(threshold=0.30)
