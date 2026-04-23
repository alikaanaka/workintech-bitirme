"""Scaling utilities."""

import pandas as pd


def passthrough_scale(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return unchanged features (LightGBM does not require scaling)."""
    return train_df, valid_df
