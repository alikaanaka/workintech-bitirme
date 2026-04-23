"""Base model interface."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract model interface for training pipeline."""

    @abstractmethod
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series) -> None:
        """Fit model on training data with validation set."""

    @abstractmethod
    def predict_proba(self, x_data: pd.DataFrame) -> np.ndarray:
        """Return probability predictions for positive class."""
