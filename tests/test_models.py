"""Tests for model wrappers."""

import pandas as pd

from src.models.lgbm_model import LGBMModel


def test_lgbm_wrapper_fit_predict_proba_shape() -> None:
    """Model wrapper should fit and output correct probability shape."""
    x_train = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "f2": [1.0, 0.0, 1.0, 0.0]})
    y_train = pd.Series([0, 1, 0, 1])
    x_valid = pd.DataFrame({"f1": [0.15, 0.35], "f2": [1.0, 0.0]})
    y_valid = pd.Series([0, 1])

    model = LGBMModel(
        params={"objective": "binary", "metric": "auc", "learning_rate": 0.1, "num_leaves": 8, "verbose": -1, "seed": 42},
        n_estimators=25,
        early_stopping_rounds=5,
        scale_pos_weight=1.0,
    )
    model.fit(x_train, y_train, x_valid, y_valid)
    proba = model.predict_proba(x_valid)

    assert proba.shape == (2,)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
