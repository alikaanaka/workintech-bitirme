"""Main table feature engineering helpers."""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _safe_divide(df: pd.DataFrame, numerator_col: str, denominator_col: str) -> pd.Series:
    """Safely divide two columns and return NaN when impossible."""
    if numerator_col not in df.columns or denominator_col not in df.columns:
        logger.info("Skipping ratio due to missing columns: %s / %s", numerator_col, denominator_col)
        return pd.Series(np.nan, index=df.index, dtype="float64")
    denominator = df[denominator_col].replace(0, np.nan)
    return df[numerator_col] / denominator


def add_main_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create core ratio and demographic features on main table."""
    featured = df.copy()
    featured["CREDIT_INCOME_RATIO"] = _safe_divide(featured, "AMT_CREDIT", "AMT_INCOME_TOTAL")
    featured["ANNUITY_INCOME_RATIO"] = _safe_divide(featured, "AMT_ANNUITY", "AMT_INCOME_TOTAL")
    featured["CREDIT_TERM"] = _safe_divide(featured, "AMT_CREDIT", "AMT_ANNUITY")
    featured["DAYS_EMPLOYED_PERCENT"] = _safe_divide(featured, "DAYS_EMPLOYED", "DAYS_BIRTH")
    featured["INCOME_PER_PERSON"] = _safe_divide(featured, "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS")

    if "DAYS_BIRTH" in featured.columns:
        featured["AGE_YEARS"] = -featured["DAYS_BIRTH"] / 365.0
    else:
        logger.info("Skipping AGE_YEARS, DAYS_BIRTH is missing")
        featured["AGE_YEARS"] = np.nan

    ext_columns = [column for column in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if column in featured.columns]
    if ext_columns:
        featured["EXT_SOURCE_MEAN"] = featured[ext_columns].mean(axis=1)
        featured["EXT_SOURCE_PROD"] = featured[ext_columns].prod(axis=1)
        featured["EXT_SOURCE_STD"] = featured[ext_columns].std(axis=1)
    else:
        logger.info("Skipping EXT_SOURCE features, no EXT_SOURCE columns found")
        featured["EXT_SOURCE_MEAN"] = np.nan
        featured["EXT_SOURCE_PROD"] = np.nan
        featured["EXT_SOURCE_STD"] = np.nan

    document_columns = [f"FLAG_DOCUMENT_{idx}" for idx in range(2, 22) if f"FLAG_DOCUMENT_{idx}" in featured.columns]
    if document_columns:
        featured["DOCUMENT_SUM"] = featured[document_columns].sum(axis=1)
    else:
        logger.info("Skipping DOCUMENT_SUM, FLAG_DOCUMENT columns are missing")
        featured["DOCUMENT_SUM"] = np.nan

    bureau_req_columns = [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]
    available_bureau_req_columns = [column for column in bureau_req_columns if column in featured.columns]
    if available_bureau_req_columns:
        featured["AMT_REQ_CREDIT_BUREAU_TOTAL"] = featured[available_bureau_req_columns].sum(axis=1)
    else:
        logger.info("Skipping AMT_REQ_CREDIT_BUREAU_TOTAL, columns are missing")
        featured["AMT_REQ_CREDIT_BUREAU_TOTAL"] = np.nan

    return featured
