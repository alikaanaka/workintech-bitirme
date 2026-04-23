"""Tests for loading, validation, and cleaning."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.cleaner import clean_dataframe
from src.data.loader import read_csv
from src.data.validator import validate_train_schema


def test_loader_reads_csv(tmp_path: Path) -> None:
    """Loader should read CSV correctly."""
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(csv_path, index=False)

    loaded = read_csv(csv_path)
    assert loaded.shape == (2, 1)


def test_validator_catches_missing_columns() -> None:
    """Validator should fail when required columns are missing."""
    df = pd.DataFrame({"SK_ID_CURR": [1, 2]})
    with pytest.raises(ValueError):
        validate_train_schema(df)


def test_cleaning_rules_apply() -> None:
    """Cleaner should apply configured rules."""
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "TARGET": [0, 1],
            "DAYS_EMPLOYED": [365243, -100],
            "DAYS_BIRTH": [-12000, 100],  # second value should become NaN due to positive DAYS_*
            "AMT_INCOME_TOTAL": [100.0, 1000000.0],
            "SELLERPLACE_AREA": [-1, 20],
            "NAME_TYPE_SUITE": ["XNA", "Family"],
        }
    )

    cleaned = clean_dataframe(df)
    assert pd.isna(cleaned.loc[0, "DAYS_EMPLOYED"])
    assert pd.isna(cleaned.loc[1, "DAYS_BIRTH"])
    assert cleaned.loc[1, "AMT_INCOME_TOTAL"] <= df["AMT_INCOME_TOTAL"].quantile(0.999)
    assert pd.isna(cleaned.loc[0, "SELLERPLACE_AREA"])
    assert pd.isna(cleaned.loc[0, "NAME_TYPE_SUITE"])
