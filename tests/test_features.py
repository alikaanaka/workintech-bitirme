"""Tests for feature generation and pipeline behavior."""

from pathlib import Path

import pandas as pd

from src.config import config
from src.features.feature_pipeline import run_feature_pipeline
from src.features.prev_aggregator import aggregate_previous_application


def _sample_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "TARGET": [0, 1, 0],
            "AMT_CREDIT": [100000.0, 200000.0, 150000.0],
            "AMT_INCOME_TOTAL": [50000.0, 100000.0, 75000.0],
            "AMT_ANNUITY": [10000.0, 20000.0, 15000.0],
            "DAYS_EMPLOYED": [-1000, -2000, -3000],
            "DAYS_BIRTH": [-12000, -15000, -18000],
            "CNT_FAM_MEMBERS": [2, 4, 3],
            "EXT_SOURCE_1": [0.2, 0.3, 0.4],
            "EXT_SOURCE_2": [0.5, 0.6, 0.7],
            "EXT_SOURCE_3": [0.8, 0.9, 0.95],
            "AMT_REQ_CREDIT_BUREAU_HOUR": [0, 1, 0],
            "AMT_REQ_CREDIT_BUREAU_DAY": [0, 0, 0],
            "AMT_REQ_CREDIT_BUREAU_WEEK": [0, 0, 1],
            "AMT_REQ_CREDIT_BUREAU_MON": [1, 0, 1],
            "AMT_REQ_CREDIT_BUREAU_QRT": [0, 0, 0],
            "AMT_REQ_CREDIT_BUREAU_YEAR": [1, 2, 0],
            "FLAG_DOCUMENT_2": [0, 1, 0],
            "FLAG_DOCUMENT_3": [1, 1, 1],
        }
    )


def _sample_prev_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 3, 3],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Canceled", "Approved", "Unused offer"],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans", "Consumer loans", "Revolving loans", "Cash loans"],
            "CODE_REJECT_REASON": ["XAP", "HC", "LIMIT", "SCO", "XNA"],
            "NAME_YIELD_GROUP": ["high", "middle", "low_normal", "high", "middle"],
            "AMT_ANNUITY": [1000.0, 1200.0, 1500.0, 1300.0, 900.0],
            "AMT_APPLICATION": [9000.0, 11000.0, 15000.0, 13000.0, 8000.0],
            "AMT_CREDIT": [10000.0, 10000.0, 14000.0, 12000.0, 7000.0],
            "AMT_DOWN_PAYMENT": [500.0, 600.0, 700.0, 300.0, 100.0],
            "AMT_GOODS_PRICE": [9500.0, 9800.0, 14000.0, 12500.0, 7900.0],
            "RATE_DOWN_PAYMENT": [0.05, 0.06, 0.07, 0.02, 0.01],
            "RATE_INTEREST_PRIMARY": [0.12, 0.14, 0.11, 0.15, 0.13],
            "RATE_INTEREST_PRIVILEGED": [0.10, 0.09, 0.08, 0.1, 0.09],
            "CNT_PAYMENT": [12, 10, 8, 6, 5],
            "DAYS_DECISION": [-200, -150, -300, -50, -20],
            "DAYS_FIRST_DUE": [-170, -120, -260, -45, -15],
            "DAYS_LAST_DUE": [-10, -5, -100, -2, -1],
            "DAYS_TERMINATION": [-5, -2, -50, -1, -1],
            "SELLERPLACE_AREA": [20, 30, 25, 15, 10],
            "NFLAG_INSURED_ON_APPROVAL": [1, 0, 1, 1, 0],
        }
    )


def test_prev_aggregator_row_count_matches_unique_customer_count() -> None:
    """Aggregator should produce one row per customer key."""
    prev_df = _sample_prev_df()
    aggregated = aggregate_previous_application(prev_df)
    assert len(aggregated) == prev_df["SK_ID_CURR"].nunique()


def test_feature_pipeline_output_contains_core_columns(tmp_path: Path) -> None:
    """Pipeline output should preserve key columns and row count."""
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = raw_dir / config.TRAIN_FILE_NAME
    prev_path = raw_dir / config.PREVIOUS_APPLICATION_FILE_NAME
    _sample_train_df().to_csv(train_path, index=False)
    _sample_prev_df().to_csv(prev_path, index=False)

    original_raw_dir = config.RAW_DATA_DIR
    original_processed_dir = config.PROCESSED_DATA_DIR
    try:
        config.RAW_DATA_DIR = raw_dir
        config.PROCESSED_DATA_DIR = processed_dir
        output_df = run_feature_pipeline()
    finally:
        config.RAW_DATA_DIR = original_raw_dir
        config.PROCESSED_DATA_DIR = original_processed_dir

    assert "SK_ID_CURR" in output_df.columns
    assert "TARGET" in output_df.columns
    assert len(output_df) == len(_sample_train_df())
    assert "PREV_APP_COUNT" in output_df.columns
    assert "CREDIT_INCOME_RATIO" in output_df.columns
