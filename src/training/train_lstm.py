"""Train LSTM with 5-fold OOF cross-validation strategy."""

from __future__ import annotations

import json
import os
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Avoid OpenMP runtime conflict between LightGBM and PyTorch in this environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from src.config import config
from src.data.cleaner import clean_dataframe
from src.data.loader import get_previous_application_path, get_train_path, read_csv
from src.evaluation.metrics import calculate_classification_metrics_with_threshold
from src.features.interaction_features import add_interaction_features
from src.features.main_features import add_main_features
from src.models.lstm_model import (
    HybridLSTMClassifier,
    SEQUENCE_CATEGORICAL_COLUMNS,
    SEQUENCE_NUMERIC_COLUMNS,
    build_sequence_dataset,
)
from src.preprocessing.encoder import FoldEncoder
from src.utils.io import ensure_directory
from src.utils.logger import get_logger

logger = get_logger(__name__)

_N_FOLDS = 5
_FINAL_MODEL_VAL_SIZE = 0.10  # internal early-stopping split for final model


# ---------------------------------------------------------------------------
# Loss & utilities
# ---------------------------------------------------------------------------

class _FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification."""

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p_t = torch.sigmoid(logits) * targets + (1.0 - torch.sigmoid(logits)) * (1.0 - targets)
        return ((1.0 - p_t) ** self.gamma * bce).mean()


def _fit_sequence_scaler(seq: np.ndarray, mask: np.ndarray) -> StandardScaler:
    """Fit StandardScaler only on real (non-padded) sequence positions."""
    flat_mask = mask.reshape(-1).astype(bool)
    flat_seq = seq.reshape(-1, seq.shape[2])
    scaler = StandardScaler()
    scaler.fit(flat_seq[flat_mask])
    return scaler


def _apply_sequence_scaler(seq: np.ndarray, mask: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Scale sequence features; zero out padded positions after scaling."""
    B, T, feat = seq.shape
    scaled = scaler.transform(seq.reshape(-1, feat)).astype(np.float32).reshape(B, T, feat)
    scaled[mask == 0] = 0.0
    return scaled


def _find_f1_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Grid-search threshold on precision-recall curve to maximise F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5, 0.0
    f1_vals = 2.0 * (precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best = int(np.argmax(f1_vals))
    return float(thresholds[best]), float(f1_vals[best])


def _select_lgbm_weight_by_f1(
    y_valid: np.ndarray,
    p_lgbm: np.ndarray,
    p_lstm: np.ndarray,
    threshold: float,
    step: float = 0.05,
) -> tuple[float, float]:
    """Grid-search LightGBM blend weight to maximise F1 at given threshold (for future stacking use)."""
    best_weight, best_f1 = 0.7, -1.0
    for w in np.arange(0.0, 1.0 + step, step):
        blended = float(w) * p_lgbm + (1.0 - float(w)) * p_lstm
        f1 = float(f1_score(y_valid, (blended >= threshold).astype(int), zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_weight = float(w)
    return best_weight, best_f1


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _prepare_static_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    train_df = read_csv(get_train_path())
    cleaned = clean_dataframe(train_df)
    featured = add_interaction_features(add_main_features(cleaned))
    y = featured[config.TARGET_COLUMN].astype(int)
    customer_ids = featured[config.KEY_COLUMN].astype(int)
    x = featured.drop(columns=[config.TARGET_COLUMN, config.KEY_COLUMN], errors="ignore")
    max_rows_raw = os.getenv(config.TRAIN_MAX_ROWS_ENV)
    if max_rows_raw:
        max_rows = int(max_rows_raw)
        if 0 < max_rows < len(x):
            sampled_idx = (
                y.groupby(y, group_keys=False)
                .apply(
                    lambda s: s.sample(
                        n=max(1, int(len(s) * max_rows / len(y))),
                        random_state=config.RANDOM_SEED,
                    )
                )
                .index
            )
            x = x.loc[sampled_idx]
            y = y.loc[sampled_idx]
            customer_ids = customer_ids.loc[sampled_idx]
            logger.warning("Sampled dataset: %s=%s", config.TRAIN_MAX_ROWS_ENV, max_rows)
    return x, y, customer_ids


def _prepare_prev_dataset(customer_ids: np.ndarray) -> pd.DataFrame:
    required_cols = [config.KEY_COLUMN, *SEQUENCE_NUMERIC_COLUMNS, *SEQUENCE_CATEGORICAL_COLUMNS]
    prev_path = get_previous_application_path()
    prev_df = pd.read_csv(prev_path, usecols=lambda col: col in required_cols, low_memory=False)
    prev_df = prev_df[prev_df[config.KEY_COLUMN].isin(set(customer_ids))].copy()
    for col in SEQUENCE_CATEGORICAL_COLUMNS:
        if col in prev_df.columns:
            prev_df[col] = prev_df[col].fillna("MISSING").astype(str).replace({"XNA": "MISSING", "XAP": "MISSING"})
    for col in SEQUENCE_NUMERIC_COLUMNS:
        if col in prev_df.columns:
            prev_df[col] = pd.to_numeric(prev_df[col], errors="coerce").fillna(0.0)
    return prev_df


# ---------------------------------------------------------------------------
# LSTM training loop
# ---------------------------------------------------------------------------

def _to_tensor_loader(
    static_x: np.ndarray,
    sequence_x: np.ndarray,
    sequence_mask: np.ndarray,
    y: np.ndarray,
    batch_size: int = 512,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(static_x, dtype=torch.float32),
        torch.tensor(sequence_x, dtype=torch.float32),
        torch.tensor(sequence_mask, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_lstm_model(
    static_train: np.ndarray,
    seq_train: np.ndarray,
    mask_train: np.ndarray,
    y_train: np.ndarray,
    static_valid: np.ndarray,
    seq_valid: np.ndarray,
    mask_valid: np.ndarray,
    y_valid: np.ndarray,
    fold_label: str = "",
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridLSTMClassifier(static_dim=static_train.shape[1], sequence_dim=seq_train.shape[2]).to(device)
    pos_count = float(np.sum(y_train == 1))
    neg_count = float(np.sum(y_train == 0))
    pos_weight = torch.tensor([(neg_count / max(pos_count, 1.0))], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    criterion = _FocalLoss(gamma=2.0, pos_weight=pos_weight)
    train_loader = _to_tensor_loader(static_train, seq_train, mask_train, y_train)
    valid_loader = _to_tensor_loader(static_valid, seq_valid, mask_valid, y_valid, shuffle=False)

    best_auc = -np.inf
    best_epoch = -1
    patience_counter = 0
    early_stopping_patience = 6
    best_state: dict[str, Any] | None = None
    best_valid_pred = np.zeros_like(y_valid, dtype=np.float32)
    max_epochs = 50
    prefix = f"[{fold_label}] " if fold_label else ""

    for epoch in range(max_epochs):
        model.train()
        train_losses: list[float] = []
        for sb, sqb, mkb, yb in train_loader:
            sb, sqb, mkb, yb = sb.to(device), sqb.to(device), mkb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(sb, sqb, mkb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        valid_logits: list[np.ndarray] = []
        valid_targets: list[np.ndarray] = []
        with torch.no_grad():
            for sb, sqb, mkb, yb in valid_loader:
                sb, sqb, mkb, yb = sb.to(device), sqb.to(device), mkb.to(device), yb.to(device)
                logits = model(sb, sqb, mkb)
                val_losses.append(float(criterion(logits, yb).item()))
                valid_logits.append(logits.detach().cpu().numpy())
                valid_targets.append(yb.detach().cpu().numpy())

        y_valid_epoch = np.concatenate(valid_targets).astype(int)
        valid_prob = torch.sigmoid(torch.tensor(np.concatenate(valid_logits))).numpy()
        valid_auc = float(roc_auc_score(y_valid_epoch, valid_prob))
        scheduler.step(valid_auc)

        logger.info(
            "%sEpoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_auc=%.6f",
            prefix, epoch + 1, max_epochs,
            float(np.mean(train_losses)),
            float(np.mean(val_losses)),
            valid_auc,
        )

        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_valid_pred = valid_prob.astype(np.float32)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("%sEarly stopping at epoch %d", prefix, epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model.cpu(),
        "best_validation_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "validation_predictions": best_valid_pred,
        "device_used": str(device),
    }


def _predict_lstm(
    model: HybridLSTMClassifier,
    static_x: np.ndarray,
    seq_x: np.ndarray,
    seq_mask: np.ndarray,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(static_x, dtype=torch.float32, device=device),
            torch.tensor(seq_x, dtype=torch.float32, device=device),
            torch.tensor(seq_mask, dtype=torch.float32, device=device),
        )
        return torch.sigmoid(logits).cpu().numpy()


# ---------------------------------------------------------------------------
# Single CV fold
# ---------------------------------------------------------------------------

def _run_cv_fold(
    fold_idx: int,
    x_fold_tr: pd.DataFrame,
    y_fold_tr: pd.Series,
    ids_fold_tr: pd.Series,
    x_fold_va: pd.DataFrame,
    y_fold_va: pd.Series,
    ids_fold_va: pd.Series,
    prev_df: pd.DataFrame,
) -> dict[str, Any]:
    """Train one LSTM fold and return OOF predictions for the validation slice."""
    label = f"Fold {fold_idx}/{_N_FOLDS}"

    encoder = FoldEncoder().fit(x_fold_tr)
    x_tr_enc = encoder.transform(x_fold_tr).fillna(0.0)
    x_va_enc = encoder.transform(x_fold_va).fillna(0.0)

    static_scaler = StandardScaler()
    x_tr_scaled = static_scaler.fit_transform(x_tr_enc.to_numpy()).astype(np.float32)
    x_va_scaled = static_scaler.transform(x_va_enc.to_numpy()).astype(np.float32)

    train_seq = build_sequence_dataset(prev_df, ids_fold_tr.to_numpy())
    valid_seq = build_sequence_dataset(prev_df, ids_fold_va.to_numpy(), categorical_maps=train_seq.categorical_maps)

    seq_scaler = _fit_sequence_scaler(train_seq.sequences, train_seq.masks)
    tr_seq_scaled = _apply_sequence_scaler(train_seq.sequences, train_seq.masks, seq_scaler)
    va_seq_scaled = _apply_sequence_scaler(valid_seq.sequences, valid_seq.masks, seq_scaler)

    outputs = _train_lstm_model(
        x_tr_scaled, tr_seq_scaled, train_seq.masks, y_fold_tr.to_numpy(),
        x_va_scaled, va_seq_scaled, valid_seq.masks, y_fold_va.to_numpy(),
        fold_label=label,
    )

    oof_scores = _predict_lstm(outputs["model"], x_va_scaled, va_seq_scaled, valid_seq.masks)
    fold_auc = float(roc_auc_score(y_fold_va.to_numpy(), oof_scores))

    logger.info(
        "%s OOF AUC=%.6f | best_val_auc=%.6f | best_epoch=%d",
        label, fold_auc, outputs["best_validation_auc"], outputs["best_epoch"],
    )

    return {
        "fold": fold_idx,
        "valid_positional_idx": np.arange(len(y_fold_va)),  # placeholder; set by caller
        "customer_ids": ids_fold_va.to_numpy(),
        "oof_scores": oof_scores,
        "oof_labels": y_fold_va.to_numpy(),
        "best_validation_auc": outputs["best_validation_auc"],
        "best_epoch": outputs["best_epoch"],
        "oof_auc": fold_auc,
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_oof_report(
    fold_results: list[dict[str, Any]],
    oof_auc_overall: float,
    oof_f1_threshold: float,
    oof_f1: float,
    holdout_metrics: dict[str, Any],
    holdout_f1_threshold: float,
    final_model_best_auc: float,
    final_model_best_epoch: int,
    device_used: str,
    model_path: Path,
    oof_artifact_path: Path,
    trained_at: str,
) -> Path:
    ensure_directory(config.REPORTS_DIR)
    report_path = config.REPORTS_DIR / f"lstm_oof_cv5_report_{trained_at}.md"

    lines = [
        "# LSTM OOF CV5 Eğitim Raporu",
        "",
        f"**Tarih:** {trained_at}  ",
        f"**Strateji:** 5-fold stratified OOF cross-validation  ",
        f"**Mimari:** HybridLSTMClassifier (BiLSTM + ScaledDotAttention + LayerNorm)  ",
        f"**Device:** {device_used}  ",
        "",
        "## Hiperparametreler",
        "",
        "| Parametre | Değer |",
        "|---|---|",
        "| hidden_size | 128 |",
        "| num_layers | 2 (bidirectional) |",
        "| dropout | 0.3 |",
        "| optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |",
        "| loss | FocalLoss (gamma=2.0, pos_weight=neg/pos) |",
        "| max_epochs | 50 |",
        "| early_stopping_patience | 6 |",
        "| gradient_clip_max_norm | 1.0 |",
        "| SEQUENCE_MAX_LEN | 15 |",
        "| batch_size | 512 |",
        "| n_folds | 5 |",
        "",
        "## Fold Bazlı Sonuçlar",
        "",
        "| Fold | OOF AUC | Best Val AUC | Best Epoch |",
        "|---:|---:|---:|---:|",
    ]
    for fr in fold_results:
        lines.append(
            f"| {fr['fold']} | {fr['oof_auc']:.6f} | {fr['best_validation_auc']:.6f} | {fr['best_epoch']} |"
        )

    mean_oof_auc = float(np.mean([fr["oof_auc"] for fr in fold_results]))
    std_oof_auc = float(np.std([fr["oof_auc"] for fr in fold_results]))
    lines.extend([
        f"| **Mean** | **{mean_oof_auc:.6f}** | — | — |",
        f"| **Std** | **{std_oof_auc:.6f}** | — | — |",
        "",
        "## OOF Genel Metrikler (tüm 5 fold birleşik)",
        "",
        f"| Metrik | Değer |",
        f"|---|---:|",
        f"| OOF AUC (overall) | {oof_auc_overall:.6f} |",
        f"| OOF F1-optimal threshold | {oof_f1_threshold:.4f} |",
        f"| OOF F1 @ optimal threshold | {oof_f1:.6f} |",
        "",
        "## Final Model — Holdout Metrikleri",
        "",
        f"Final model tüm train seti üzerinde eğitildi (erken durdurma için %10 iç val ayrıldı).  ",
        f"**Best epoch:** {final_model_best_epoch} | **Best val AUC (iç):** {final_model_best_auc:.6f}  ",
        "",
        "| Metrik | Değer |",
        "|---|---:|",
        f"| Holdout AUC | {holdout_metrics['auc_roc']:.6f} |",
        f"| Holdout Gini | {holdout_metrics['gini']:.6f} |",
        f"| Holdout KS | {holdout_metrics['ks_statistic']:.6f} |",
        f"| Holdout PR-AUC | {holdout_metrics['pr_auc']:.6f} |",
        f"| Holdout F1 @ {holdout_f1_threshold:.4f} | {holdout_metrics['f1_at_threshold']:.6f} |",
        f"| Holdout Brier Score | {holdout_metrics['brier_score']:.6f} |",
        "",
        f"**Confusion Matrix (holdout):** `{holdout_metrics['confusion_matrix']}`",
        "",
        "## Kaydedilen Dosyalar",
        "",
        f"- Model: `{model_path}`",
        f"- OOF predictions: `{oof_artifact_path}`",
        "",
        "## Teknik Notlar",
        "",
        "- OOF tahminleri her fold'da o fold'un kendi encoder/scaler/categorical_maps'i ile üretildi.",
        "- Final model, tüm train_full verisi üzerinde bağımsız olarak eğitildi.",
        "- OOF tahminleri stacking meta-model için kullanıma hazır.",
        "- Geçmişsiz müşteri: mask tamamen sıfır → _ScaledDotAttention uniform attention → static feature tabanlı tahmin.",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_lstm_oof_cv5(threshold: float = 0.30) -> dict[str, Any]:
    """Train LSTM with 5-fold OOF strategy. Saves model + OOF predictions."""
    _set_reproducible_seed(config.RANDOM_SEED)

    x_static, y, customer_ids = _prepare_static_dataset()

    # Holdout split (same seed as before for reproducibility)
    x_train_full, x_holdout, y_train_full, y_holdout, ids_train_full, ids_holdout = train_test_split(
        x_static, y, customer_ids,
        test_size=config.HOLDOUT_TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )

    # Load previous application data for all relevant customers
    all_ids = np.concatenate([ids_train_full.to_numpy(), ids_holdout.to_numpy()])
    prev_df = _prepare_prev_dataset(all_ids)

    # -----------------------------------------------------------------------
    # 5-fold OOF cross-validation
    # -----------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    oof_scores = np.zeros(len(y_train_full), dtype=np.float32)
    oof_customer_ids = np.zeros(len(y_train_full), dtype=np.int64)
    fold_results: list[dict[str, Any]] = []

    x_tf_reset = x_train_full.reset_index(drop=True)
    y_tf_reset = y_train_full.reset_index(drop=True)
    ids_tf_reset = ids_train_full.reset_index(drop=True)

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(x_tf_reset, y_tf_reset), start=1):
        fr = _run_cv_fold(
            fold_idx=fold_idx,
            x_fold_tr=x_tf_reset.iloc[train_idx],
            y_fold_tr=y_tf_reset.iloc[train_idx],
            ids_fold_tr=ids_tf_reset.iloc[train_idx],
            x_fold_va=x_tf_reset.iloc[valid_idx],
            y_fold_va=y_tf_reset.iloc[valid_idx],
            ids_fold_va=ids_tf_reset.iloc[valid_idx],
            prev_df=prev_df,
        )
        oof_scores[valid_idx] = fr["oof_scores"]
        oof_customer_ids[valid_idx] = fr["customer_ids"]
        fold_results.append(fr)

    oof_auc_overall = float(roc_auc_score(y_tf_reset.to_numpy(), oof_scores))
    oof_f1_threshold, oof_f1 = _find_f1_optimal_threshold(y_tf_reset.to_numpy(), oof_scores)
    oof_f1_threshold = float(np.clip(oof_f1_threshold, 0.05, 0.75))
    logger.info(
        "OOF overall AUC=%.6f | F1-optimal threshold=%.4f | OOF F1=%.6f",
        oof_auc_overall, oof_f1_threshold, oof_f1,
    )

    # -----------------------------------------------------------------------
    # Final model — trained on all train_full with a small internal val split
    # -----------------------------------------------------------------------
    logger.info("Training final model on full train split...")
    x_fin_tr, x_fin_va, y_fin_tr, y_fin_va, ids_fin_tr, ids_fin_va = train_test_split(
        x_tf_reset, y_tf_reset, ids_tf_reset,
        test_size=_FINAL_MODEL_VAL_SIZE,
        stratify=y_tf_reset,
        random_state=config.RANDOM_SEED,
    )

    final_encoder = FoldEncoder().fit(x_fin_tr)
    x_fin_tr_enc = final_encoder.transform(x_fin_tr).fillna(0.0)
    x_fin_va_enc = final_encoder.transform(x_fin_va).fillna(0.0)
    x_holdout_enc = final_encoder.transform(x_holdout).fillna(0.0)

    final_static_scaler = StandardScaler()
    x_fin_tr_scaled = final_static_scaler.fit_transform(x_fin_tr_enc.to_numpy()).astype(np.float32)
    x_fin_va_scaled = final_static_scaler.transform(x_fin_va_enc.to_numpy()).astype(np.float32)
    x_holdout_scaled = final_static_scaler.transform(x_holdout_enc.to_numpy()).astype(np.float32)

    final_train_seq = build_sequence_dataset(prev_df, ids_fin_tr.to_numpy())
    final_val_seq = build_sequence_dataset(prev_df, ids_fin_va.to_numpy(), categorical_maps=final_train_seq.categorical_maps)
    holdout_seq = build_sequence_dataset(prev_df, ids_holdout.to_numpy(), categorical_maps=final_train_seq.categorical_maps)

    final_seq_scaler = _fit_sequence_scaler(final_train_seq.sequences, final_train_seq.masks)
    fin_tr_seq_scaled = _apply_sequence_scaler(final_train_seq.sequences, final_train_seq.masks, final_seq_scaler)
    fin_va_seq_scaled = _apply_sequence_scaler(final_val_seq.sequences, final_val_seq.masks, final_seq_scaler)
    holdout_seq_scaled = _apply_sequence_scaler(holdout_seq.sequences, holdout_seq.masks, final_seq_scaler)

    final_outputs = _train_lstm_model(
        x_fin_tr_scaled,
        fin_tr_seq_scaled,
        final_train_seq.masks,
        y_fin_tr.to_numpy(),
        x_fin_va_scaled,
        fin_va_seq_scaled,
        final_val_seq.masks,
        y_fin_va.to_numpy(),
        fold_label="Final",
    )

    final_model = final_outputs["model"]
    holdout_score = _predict_lstm(final_model, x_holdout_scaled, holdout_seq_scaled, holdout_seq.masks)
    holdout_f1_threshold, _ = _find_f1_optimal_threshold(y_holdout.to_numpy(), holdout_score)
    holdout_f1_threshold = float(np.clip(holdout_f1_threshold, 0.05, 0.75))
    holdout_metrics = calculate_classification_metrics_with_threshold(
        y_holdout.to_numpy(), holdout_score, holdout_f1_threshold
    )
    logger.info(
        "Final model holdout: AUC=%.6f | F1=%.6f @ threshold=%.4f",
        holdout_metrics["auc_roc"], holdout_metrics["f1_at_threshold"], holdout_f1_threshold,
    )

    # -----------------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------------
    trained_at = datetime.now().strftime(config.MODEL_DATE_FORMAT)
    ensure_directory(config.MODELS_SAVED_DIR)

    model_save_path = config.MODELS_SAVED_DIR / f"lstm_oof_cv5_{trained_at}.pt"
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "static_dim": int(x_fin_tr_enc.shape[1]),
            "sequence_dim": int(final_train_seq.sequences.shape[2]),
            "categorical_maps": final_train_seq.categorical_maps,
            "f1_threshold": float(holdout_f1_threshold),
            "oof_f1_threshold": float(oof_f1_threshold),
            "best_validation_auc": float(final_outputs["best_validation_auc"]),
            "best_epoch": int(final_outputs["best_epoch"]),
            "static_scaler": final_static_scaler,
            "seq_scaler": final_seq_scaler,
            "encoder": final_encoder,
            "trained_at": trained_at,
            "strategy": "oof_cv5",
        },
        model_save_path,
    )

    oof_artifact_path = config.MODELS_SAVED_DIR / f"lstm_oof_predictions_{trained_at}.npz"
    np.savez(
        oof_artifact_path,
        customer_ids=oof_customer_ids,
        oof_scores=oof_scores,
        y_true=y_tf_reset.to_numpy().astype(np.int32),
    )

    holdout_artifact_path = config.MODELS_SAVED_DIR / f"lstm_holdout_predictions_{trained_at}.npz"
    np.savez(
        holdout_artifact_path,
        customer_ids=ids_holdout.to_numpy(),
        holdout_scores=holdout_score.astype(np.float32),
        y_true=y_holdout.to_numpy().astype(np.int32),
    )

    metadata = {
        "strategy": "oof_cv5",
        "n_folds": _N_FOLDS,
        "trained_at": trained_at,
        "oof_auc_overall": float(oof_auc_overall),
        "oof_f1_threshold": float(oof_f1_threshold),
        "oof_f1": float(oof_f1),
        "holdout_auc": float(holdout_metrics["auc_roc"]),
        "holdout_f1_threshold": float(holdout_f1_threshold),
        "holdout_f1": float(holdout_metrics["f1_at_threshold"]),
        "final_model_best_epoch": int(final_outputs["best_epoch"]),
        "final_model_best_val_auc": float(final_outputs["best_validation_auc"]),
        "fold_results": [
            {
                "fold": fr["fold"],
                "oof_auc": float(fr["oof_auc"]),
                "best_validation_auc": float(fr["best_validation_auc"]),
                "best_epoch": int(fr["best_epoch"]),
            }
            for fr in fold_results
        ],
        "artifacts": {
            "model": str(model_save_path.relative_to(config.PROJECT_ROOT)),
            "oof_predictions": str(oof_artifact_path.relative_to(config.PROJECT_ROOT)),
            "holdout_predictions": str(holdout_artifact_path.relative_to(config.PROJECT_ROOT)),
        },
    }
    (config.MODELS_SAVED_DIR / "lstm_oof_cv5_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    report_path = _write_oof_report(
        fold_results=fold_results,
        oof_auc_overall=oof_auc_overall,
        oof_f1_threshold=oof_f1_threshold,
        oof_f1=oof_f1,
        holdout_metrics=holdout_metrics,
        holdout_f1_threshold=holdout_f1_threshold,
        final_model_best_auc=float(final_outputs["best_validation_auc"]),
        final_model_best_epoch=int(final_outputs["best_epoch"]),
        device_used=final_outputs["device_used"],
        model_path=model_save_path,
        oof_artifact_path=oof_artifact_path,
        trained_at=trained_at,
    )
    logger.info("OOF CV5 report written: %s", report_path)

    return {
        "oof_auc_overall": float(oof_auc_overall),
        "oof_f1_threshold": float(oof_f1_threshold),
        "oof_f1": float(oof_f1),
        "holdout_metrics": holdout_metrics,
        "holdout_f1_threshold": float(holdout_f1_threshold),
        "fold_results": fold_results,
        "model_path": str(model_save_path),
        "oof_artifact_path": str(oof_artifact_path),
        "holdout_artifact_path": str(holdout_artifact_path),
        "report_path": str(report_path),
        "trained_at": trained_at,
    }


# Backward-compatible alias (for code that may still import the old name)
def train_lstm_and_ensemble(threshold: float = 0.30) -> dict[str, Any]:
    return train_lstm_oof_cv5(threshold=threshold)


if __name__ == "__main__":
    train_lstm_oof_cv5(threshold=0.30)
