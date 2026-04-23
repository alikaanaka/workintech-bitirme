"""End-to-end feature pipeline for Phase 1 preprocessing."""

from pathlib import Path

import pandas as pd

from src.config import config
from src.data.cleaner import clean_dataframe
from src.data.loader import get_previous_application_path, get_train_path, read_csv
from src.data.validator import validate_previous_schema, validate_train_schema
from src.features.interaction_features import add_interaction_features
from src.features.main_features import add_main_features
from src.features.prev_aggregator import aggregate_previous_application
from src.utils.io import ensure_directory
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _safe_left_merge(main_df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
    """Left join while resolving duplicated column names deterministically."""
    overlapping = [column for column in prev_df.columns if column in main_df.columns and column != config.KEY_COLUMN]
    if overlapping:
        prev_df = prev_df.rename(columns={column: f"{column}_PREV" for column in overlapping})
    return main_df.merge(prev_df, on=config.KEY_COLUMN, how="left")


def run_feature_pipeline(output_path: Path | None = None) -> pd.DataFrame:
    """Run full preprocessing and feature generation pipeline."""
    train_path = get_train_path()
    prev_path = get_previous_application_path()
    logger.info("Reading train dataset from %s", train_path)
    train_df = read_csv(train_path)
    logger.info("Reading previous application dataset from %s", prev_path)
    prev_df = read_csv(prev_path)

    validate_train_schema(train_df, strict_key_uniqueness=False)
    validate_previous_schema(prev_df)

    train_clean = clean_dataframe(train_df)
    prev_clean = clean_dataframe(prev_df)

    main_features_df = add_main_features(train_clean)
    main_features_df = add_interaction_features(main_features_df)
    prev_agg_df = aggregate_previous_application(prev_clean)

    final_df = _safe_left_merge(main_features_df, prev_agg_df)
    final_df["NO_PREV_APP_FLAG"] = final_df["PREV_APP_COUNT"].isna().astype(int)

    if len(final_df) != len(train_df):
        raise ValueError("Join changed row count for main table, expected left join semantics")

    output = output_path or (config.PROCESSED_DATA_DIR / config.OUTPUT_FILE_NAME)
    ensure_directory(config.PROCESSED_DATA_DIR)
    final_df.to_parquet(output, index=False)
    logger.info(
        "Saved final dataset to %s | rows=%s cols=%s",
        output,
        len(final_df),
        len(final_df.columns),
    )
    return final_df
