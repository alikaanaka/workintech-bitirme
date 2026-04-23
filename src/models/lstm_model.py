"""LSTM model and sequence preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import config

SEQUENCE_MAX_LEN = 15
SEQUENCE_NUMERIC_COLUMNS = ["AMT_CREDIT", "AMT_APPLICATION", "AMT_ANNUITY", "CNT_PAYMENT", "DAYS_DECISION"]
SEQUENCE_CATEGORICAL_COLUMNS = ["NAME_CONTRACT_TYPE", "NAME_CONTRACT_STATUS"]


def _safe_numeric(value: Any) -> float:
    """Convert nullable values to finite float."""
    if pd.isna(value):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class SequenceArtifacts:
    """Prepared sequence data artifacts."""

    sequences: np.ndarray
    masks: np.ndarray
    customer_ids: np.ndarray
    categorical_maps: dict[str, dict[str, int]]


def build_sequence_dataset(
    prev_df: pd.DataFrame,
    customer_ids: np.ndarray,
    max_len: int = SEQUENCE_MAX_LEN,
    categorical_maps: dict[str, dict[str, int]] | None = None,
) -> SequenceArtifacts:
    """Build deterministic padded sequences per customer."""
    working = prev_df.copy()
    if "DAYS_DECISION" in working.columns:
        working = working.sort_values([config.KEY_COLUMN, "DAYS_DECISION"], ascending=[True, False])
    else:
        working = working.sort_values([config.KEY_COLUMN], ascending=[True])

    for column in SEQUENCE_NUMERIC_COLUMNS:
        if column not in working.columns:
            working[column] = 0.0
    for column in SEQUENCE_CATEGORICAL_COLUMNS:
        if column not in working.columns:
            working[column] = "MISSING"

    if categorical_maps is None:
        categorical_maps = {}
        for column in SEQUENCE_CATEGORICAL_COLUMNS:
            values = working[column].fillna("MISSING").astype(str).unique().tolist()
            categorical_maps[column] = {value: idx + 1 for idx, value in enumerate(sorted(values))}

    feature_dim = len(SEQUENCE_NUMERIC_COLUMNS) + len(SEQUENCE_CATEGORICAL_COLUMNS)
    sequences = np.zeros((len(customer_ids), max_len, feature_dim), dtype=np.float32)
    masks = np.zeros((len(customer_ids), max_len), dtype=np.float32)

    grouped = {customer_id: frame for customer_id, frame in working.groupby(config.KEY_COLUMN, sort=False)}
    for row_idx, customer_id in enumerate(customer_ids):
        customer_rows = grouped.get(customer_id)
        if customer_rows is None or customer_rows.empty:
            continue
        tail = customer_rows.head(max_len)
        for seq_idx, (_, event) in enumerate(tail.iterrows()):
            features: list[float] = []
            for column in SEQUENCE_NUMERIC_COLUMNS:
                features.append(_safe_numeric(event[column]))
            for column in SEQUENCE_CATEGORICAL_COLUMNS:
                token = str(event[column]) if not pd.isna(event[column]) else "MISSING"
                features.append(float(categorical_maps[column].get(token, 0)))
            sequences[row_idx, seq_idx, :] = np.asarray(features, dtype=np.float32)
            masks[row_idx, seq_idx] = 1.0

    return SequenceArtifacts(
        sequences=sequences,
        masks=masks,
        customer_ids=np.asarray(customer_ids),
        categorical_maps=categorical_maps,
    )


class _ScaledDotAttention(nn.Module):
    """Learnable query attention over LSTM timesteps."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # lstm_out: (B, T, H)   mask: (B, T)
        scores = self.query(lstm_out).squeeze(-1)          # (B, T)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        # Fall back to uniform if all positions masked (no prev apps)
        all_masked = (mask.sum(dim=1) == 0).unsqueeze(1)  # (B, 1)
        safe_scores = scores.masked_fill(all_masked, 0.0)
        weights = torch.softmax(safe_scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (lstm_out * weights).sum(dim=1)             # (B, H)


class HybridLSTMClassifier(nn.Module):
    """Static + sequence hybrid classifier with attention pooling."""

    def __init__(
        self,
        static_dim: int,
        sequence_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.sequence_proj = nn.Sequential(
            nn.Linear(sequence_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.attention = _ScaledDotAttention(hidden_size * 2)
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, static_x: torch.Tensor, sequence_x: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        projected = self.sequence_proj(sequence_x)
        lstm_out, _ = self.lstm(projected)
        pooled = self.attention(lstm_out, sequence_mask)   # (B, hidden*2)
        static_feat = self.static_proj(static_x)
        combined = torch.cat([pooled, static_feat], dim=1)
        logits = self.head(combined).squeeze(1)
        return logits
