"""Project configuration constants for preprocessing and training."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE_NAME = "train_merged.csv"
PREVIOUS_APPLICATION_FILE_NAME = "previous_application.csv"
OUTPUT_FILE_NAME = "final_dataset.parquet"
TOP50_FEATURES_FILE_NAME = "top50_features.json"
METRICS_REPORT_FILE_NAME = "metrics_comparison.md"
METRICS_V2_REPORT_FILE_NAME = "metrics_comparison_v2.md"
MODEL_FILE_NAME = "lgbm_model.pkl"
FEATURE_LISTS_DIR = ARTIFACTS_DIR / "feature_lists"
MODELS_SAVED_DIR = PROJECT_ROOT / "models_saved"

# Backward compatible fallback names for this repository.
TRAIN_FALLBACK_FILE_NAMES = ("train_feature.csv",)
PREVIOUS_FALLBACK_FILE_NAMES = ("prev_app_customer_level.csv",)

TARGET_COLUMN = "TARGET"
KEY_COLUMN = "SK_ID_CURR"
RANDOM_SEED = 42

INCOME_WINSORIZE_QUANTILE = 0.999
DAYS_EMPLOYED_ANOMALY_VALUE = 365243
SELLERPLACE_AREA_MISSING_VALUE = -1
UNKNOWN_CATEGORICAL_VALUES = {"XNA", "XAP"}

REQUIRED_TRAIN_COLUMNS = [KEY_COLUMN, TARGET_COLUMN]
REQUIRED_PREV_COLUMNS = [KEY_COLUMN]

PREV_STATUS_COLUMN = "NAME_CONTRACT_STATUS"
PREV_REJECT_REASON_COLUMN = "CODE_REJECT_REASON"
PREV_YIELD_GROUP_COLUMN = "NAME_YIELD_GROUP"
PREV_CONTRACT_TYPE_COLUMN = "NAME_CONTRACT_TYPE"

PREV_NUMERIC_AGG_COLUMNS = [
    "AMT_ANNUITY",
    "AMT_APPLICATION",
    "AMT_CREDIT",
    "AMT_DOWN_PAYMENT",
    "AMT_GOODS_PRICE",
    "RATE_DOWN_PAYMENT",
    "RATE_INTEREST_PRIMARY",
    "RATE_INTEREST_PRIVILEGED",
    "CNT_PAYMENT",
    "DAYS_DECISION",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
    "SELLERPLACE_AREA",
]

PREV_RATIO_SOURCE_COLUMNS = {
    "credit_to_application_num": "AMT_CREDIT",
    "credit_to_application_den": "AMT_APPLICATION",
    "down_payment_num": "AMT_DOWN_PAYMENT",
    "down_payment_den": "AMT_CREDIT",
    "insured_flag": "NFLAG_INSURED_ON_APPROVAL",
}

HOLDOUT_TEST_SIZE = 0.2
VALIDATION_SIZE_IN_TRAIN = 0.2

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "min_child_samples": 100,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": RANDOM_SEED,
    "verbose": -1,
    "n_jobs": -1,
}

LGBM_NUM_BOOST_ROUND = 1000
LGBM_EARLY_STOPPING_ROUNDS = 50
CLASSIFICATION_THRESHOLD = 0.22
TOP_FEATURE_COUNT = 50
TRAIN_MAX_ROWS_ENV = "HC_TRAIN_MAX_ROWS"

LGBM_V2_DEFAULT_THRESHOLD = 0.30
MODEL_VERSION_V2 = "v2"
MODEL_DATE_FORMAT = "%Y%m%d"
