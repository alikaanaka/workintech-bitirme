"""Data split helpers."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import config


def stratified_holdout_split(
    features: pd.DataFrame,
    target: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create stratified train/holdout split."""
    return train_test_split(
        features,
        target,
        test_size=config.HOLDOUT_TEST_SIZE,
        stratify=target,
        random_state=config.RANDOM_SEED,
    )


def stratified_train_validation_split(
    x_train_full: pd.DataFrame,
    y_train_full: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split train set into train/validation subsets with stratification."""
    return train_test_split(
        x_train_full,
        y_train_full,
        test_size=config.VALIDATION_SIZE_IN_TRAIN,
        stratify=y_train_full,
        random_state=config.RANDOM_SEED,
    )
