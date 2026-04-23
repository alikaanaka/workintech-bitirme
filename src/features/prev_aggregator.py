"""Aggregation logic for previous_application table."""

import numpy as np
import pandas as pd

from src.config import config


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return safe ratio by replacing 0 denominator with NaN."""
    return numerator / denominator.replace(0, np.nan)


def _most_common_or_nan(values: pd.Series) -> object:
    """Return mode value if available."""
    modes = values.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else np.nan


def aggregate_previous_application(prev_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous application rows to single row per SK_ID_CURR."""
    if config.KEY_COLUMN not in prev_df.columns:
        raise ValueError("Cannot aggregate previous application: SK_ID_CURR is missing")

    working = prev_df.copy()
    grouped = working.groupby(config.KEY_COLUMN, dropna=False)
    aggregated = pd.DataFrame(index=grouped.size().index)

    aggregated["PREV_APP_COUNT"] = grouped.size()
    if config.PREV_STATUS_COLUMN in working.columns:
        status_pivot = (
            working.groupby([config.KEY_COLUMN, config.PREV_STATUS_COLUMN]).size().unstack(fill_value=0)
        )
        aggregated["PREV_APP_APPROVED_COUNT"] = status_pivot.get("Approved", 0)
        aggregated["PREV_APP_REFUSED_COUNT"] = status_pivot.get("Refused", 0)
        aggregated["PREV_APP_CANCELED_COUNT"] = status_pivot.get("Canceled", 0)
        aggregated["PREV_APP_UNUSED_COUNT"] = status_pivot.get("Unused offer", 0)
    else:
        aggregated["PREV_APP_APPROVED_COUNT"] = 0
        aggregated["PREV_APP_REFUSED_COUNT"] = 0
        aggregated["PREV_APP_CANCELED_COUNT"] = 0
        aggregated["PREV_APP_UNUSED_COUNT"] = 0
    aggregated["PREV_APP_APPROVAL_RATE"] = _safe_ratio(
        aggregated["PREV_APP_APPROVED_COUNT"],
        aggregated["PREV_APP_COUNT"],
    )
    aggregated["PREV_APP_REFUSAL_RATE"] = _safe_ratio(
        aggregated["PREV_APP_REFUSED_COUNT"],
        aggregated["PREV_APP_COUNT"],
    )

    numeric_columns = [column for column in config.PREV_NUMERIC_AGG_COLUMNS if column in working.columns]
    if numeric_columns:
        numeric_agg = grouped[numeric_columns].agg(["mean", "min", "max", "sum", "std"])
        numeric_agg.columns = [f"PREV_{column}_{stat}".upper() for column, stat in numeric_agg.columns]
        aggregated = aggregated.join(numeric_agg, how="left")

    ratio_num_col = config.PREV_RATIO_SOURCE_COLUMNS["credit_to_application_num"]
    ratio_den_col = config.PREV_RATIO_SOURCE_COLUMNS["credit_to_application_den"]
    if ratio_num_col in working.columns and ratio_den_col in working.columns:
        ratio_series = _safe_ratio(working[ratio_num_col], working[ratio_den_col])
        aggregated["PREV_CREDIT_TO_APPLICATION_RATIO"] = ratio_series.groupby(working[config.KEY_COLUMN]).mean()
    else:
        aggregated["PREV_CREDIT_TO_APPLICATION_RATIO"] = np.nan

    down_num_col = config.PREV_RATIO_SOURCE_COLUMNS["down_payment_num"]
    down_den_col = config.PREV_RATIO_SOURCE_COLUMNS["down_payment_den"]
    if down_num_col in working.columns and down_den_col in working.columns:
        down_ratio = _safe_ratio(working[down_num_col], working[down_den_col])
        aggregated["PREV_DOWN_PAYMENT_RATIO_MEAN"] = down_ratio.groupby(working[config.KEY_COLUMN]).mean()
    else:
        aggregated["PREV_DOWN_PAYMENT_RATIO_MEAN"] = np.nan

    insured_col = config.PREV_RATIO_SOURCE_COLUMNS["insured_flag"]
    if insured_col in working.columns:
        aggregated["PREV_INSURED_RATIO"] = grouped[insured_col].mean()
    else:
        aggregated["PREV_INSURED_RATIO"] = np.nan

    if "DAYS_DECISION" in working.columns:
        aggregated["PREV_DAYS_DECISION_MIN"] = grouped["DAYS_DECISION"].min()
        aggregated["PREV_LAST_APPLICATION_RECENCY"] = -grouped["DAYS_DECISION"].max()
    else:
        aggregated["PREV_DAYS_DECISION_MIN"] = np.nan
        aggregated["PREV_LAST_APPLICATION_RECENCY"] = np.nan

    if config.PREV_STATUS_COLUMN in working.columns:
        active_mask = working[config.PREV_STATUS_COLUMN].isin(["Approved", "Active"]).astype(int)
        aggregated["PREV_ACTIVE_LOANS_COUNT"] = active_mask.groupby(working[config.KEY_COLUMN]).sum()
    else:
        aggregated["PREV_ACTIVE_LOANS_COUNT"] = 0

    if config.PREV_CONTRACT_TYPE_COLUMN in working.columns:
        aggregated["PREV_CONTRACT_TYPE_MOST_COMMON"] = grouped[config.PREV_CONTRACT_TYPE_COLUMN].apply(_most_common_or_nan)
    else:
        aggregated["PREV_CONTRACT_TYPE_MOST_COMMON"] = np.nan

    reject_col = config.PREV_REJECT_REASON_COLUMN
    reject_counts = {
        "PREV_REJECT_REASON_HC_COUNT": "HC",
        "PREV_REJECT_REASON_LIMIT_COUNT": "LIMIT",
        "PREV_REJECT_REASON_SCO_COUNT": "SCO",
        "PREV_REJECT_REASON_XAP_COUNT": "XAP",
        "PREV_REJECT_REASON_XNA_COUNT": "XNA",
    }
    if reject_col in working.columns:
        reject_pivot = working.groupby([config.KEY_COLUMN, reject_col]).size().unstack(fill_value=0)
        for feature_name, code in reject_counts.items():
            aggregated[feature_name] = reject_pivot.get(code, 0)
    else:
        for feature_name in reject_counts:
            aggregated[feature_name] = 0

    if config.PREV_YIELD_GROUP_COLUMN in working.columns:
        high_mask = (working[config.PREV_YIELD_GROUP_COLUMN] == "high").astype(float)
        low_norm_mask = working[config.PREV_YIELD_GROUP_COLUMN].isin(["low_normal", "middle"]).astype(float)
        aggregated["PREV_YIELD_GROUP_HIGH_RATIO"] = high_mask.groupby(working[config.KEY_COLUMN]).mean()
        aggregated["PREV_YIELD_GROUP_LOW_NORMAL_RATIO"] = low_norm_mask.groupby(working[config.KEY_COLUMN]).mean()
    else:
        aggregated["PREV_YIELD_GROUP_HIGH_RATIO"] = np.nan
        aggregated["PREV_YIELD_GROUP_LOW_NORMAL_RATIO"] = np.nan

    output = aggregated.reset_index()
    return output
