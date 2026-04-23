"""LightGBM model wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import config
from src.models.base_model import BaseModel


@dataclass
class LGBMFitResult:
    """Training summary for a single fit call."""

    best_iteration: int


class LGBMModel(BaseModel):
    """Thin wrapper around LightGBM classifier."""

    def __init__(
        self,
        params: dict | None = None,
        n_estimators: int | None = None,
        early_stopping_rounds: int | None = None,
        scale_pos_weight: float = 1.0,
    ) -> None:
        self.params = dict(config.LGBM_PARAMS if params is None else params)
        self.params["scale_pos_weight"] = scale_pos_weight
        self.n_estimators = n_estimators or config.LGBM_NUM_BOOST_ROUND
        self.early_stopping_rounds = early_stopping_rounds or config.LGBM_EARLY_STOPPING_ROUNDS
        self.model = lgb.LGBMClassifier(**self.params, n_estimators=self.n_estimators)
        self.fit_result = LGBMFitResult(best_iteration=self.n_estimators)

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series) -> None:
        """Fit model and track best iteration."""
        self.model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )
        best_iteration = self.model.best_iteration_ or self.n_estimators
        self.fit_result = LGBMFitResult(best_iteration=int(best_iteration))

    def predict_proba(self, x_data: pd.DataFrame) -> np.ndarray:
        """Predict positive class probability."""
        return self.model.predict_proba(x_data)[:, 1]
