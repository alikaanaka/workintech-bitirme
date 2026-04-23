"""Tests for sequence pipeline."""

import numpy as np
import pandas as pd

from src.models.lstm_model import build_sequence_dataset


def test_sequence_builder_is_deterministic_and_padded() -> None:
    prev = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "DAYS_DECISION": [-5, -10, -7],
            "AMT_CREDIT": [1000, 2000, 500],
            "AMT_APPLICATION": [900, 1900, 450],
            "AMT_ANNUITY": [100, 200, 50],
            "CNT_PAYMENT": [12, 18, 6],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Consumer loans", "Cash loans"],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
        }
    )
    customer_ids = np.array([1, 2, 3])
    seq = build_sequence_dataset(prev, customer_ids, max_len=3)
    assert seq.sequences.shape == (3, 3, 7)
    assert seq.masks.shape == (3, 3)
    assert seq.masks[0].sum() == 2
    assert seq.masks[1].sum() == 1
    assert seq.masks[2].sum() == 0
