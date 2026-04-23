"""Calibration utilities."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """Compute expected calibration error."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.digitize(y_prob, edges, right=True)
    ece = 0.0
    total = len(y_true)
    for bucket_id in range(1, bins + 1):
        mask = bucket == bucket_id
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += (np.sum(mask) / total) * abs(acc - conf)
    return float(ece)
