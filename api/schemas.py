"""Pydantic v2 request / response models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PreviousApplication(BaseModel):
    """One entry in a customer's previous application history (sequence input for LSTM)."""

    AMT_CREDIT: Optional[float] = None
    AMT_APPLICATION: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    CNT_PAYMENT: Optional[float] = None
    DAYS_DECISION: Optional[float] = None
    NAME_CONTRACT_TYPE: Optional[str] = None
    NAME_CONTRACT_STATUS: Optional[str] = None


class PredictRequest(BaseModel):
    """Prediction request — top-50 LGBM features (post-encoding) + optional sequence."""

    # External risk scores
    EXT_SOURCE_MEAN: Optional[float] = None
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    EXT_SOURCE_PROD: Optional[float] = None
    EXT_SOURCE_STD: Optional[float] = None

    # Loan / credit ratios
    CREDIT_TERM: Optional[float] = None
    ANNUITY_INCOME_RATIO: Optional[float] = None
    CREDIT_INCOME_RATIO: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    AMT_CREDIT: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    AMT_INCOME_TOTAL: Optional[float] = None
    INCOME_PER_PERSON: Optional[float] = None

    # Days features
    DAYS_EMPLOYED: Optional[float] = None
    DAYS_EMPLOYED_PERCENT: Optional[float] = None
    DAYS_BIRTH: Optional[float] = None
    DAYS_ID_PUBLISH: Optional[float] = None
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    DAYS_REGISTRATION: Optional[float] = None

    # Demographic
    AGE_YEARS: Optional[float] = None
    OWN_CAR_AGE: Optional[float] = None
    REGION_POPULATION_RELATIVE: Optional[float] = None
    TOTALAREA_MODE: Optional[float] = None
    BASEMENTAREA_MODE: Optional[float] = None

    # Bureau features (b_)
    b_avg_utilization: Optional[float] = None
    b_avg_loan_duration: Optional[float] = None
    b_total_current_debt: Optional[float] = None
    b_total_history_months: Optional[float] = None
    b_total_loan_count: Optional[float] = None
    b_active_loan_count: Optional[float] = None
    b_closed_loan_count: Optional[float] = None

    # Installment features (int_)
    int_max_ins_days_late_ever: Optional[float] = None
    int_total_remaining_installments: Optional[float] = None
    int_avg_payment_performance: Optional[float] = None
    int_total_remaining_debt: Optional[float] = None
    int_total_prev_loans_count: Optional[float] = None
    int_max_pos_dpd_ever: Optional[float] = None

    # Credit card features (cc_)
    cc_total_avg_utilization_ratio: Optional[float] = None
    cc_total_credit_card_experience_months: Optional[float] = None
    cc_avg_repayment_performance: Optional[float] = None
    cc_total_transaction_count: Optional[float] = None
    cc_max_balance_ever: Optional[float] = None
    cc_total_current_debt: Optional[float] = None

    # Encoded categorical features (integer codes)
    ORGANIZATION_TYPE_ENC: Optional[float] = None
    NAME_EDUCATION_TYPE_ENC: Optional[float] = None
    OCCUPATION_TYPE_ENC: Optional[float] = None
    CODE_GENDER_ENC: Optional[float] = None
    WEEKDAY_APPR_PROCESS_START_ENC: Optional[float] = None

    # Application timing
    HOUR_APPR_PROCESS_START: Optional[float] = None

    # Sequence history (LSTM input)
    previous_applications: Optional[list[PreviousApplication]] = Field(default=None)

    model_config = {"extra": "ignore"}


class FieldValidationError(BaseModel):
    field: str
    message: str


class PredictResponse(BaseModel):
    model_version: str
    request_id: str
    inference_time_ms: float
    proba_lgbm: Optional[float]
    proba_lstm: Optional[float]
    proba_ensemble: Optional[float]
    available_models: list[str]
    risk_band: str
    risk_score_pct: int
    decision: str
    threshold_used: float
    warnings: list[str]


class ExplainRequest(BaseModel):
    """Same input as PredictRequest but returns feature contributions."""

    EXT_SOURCE_MEAN: Optional[float] = None
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    EXT_SOURCE_PROD: Optional[float] = None
    EXT_SOURCE_STD: Optional[float] = None
    CREDIT_TERM: Optional[float] = None
    ANNUITY_INCOME_RATIO: Optional[float] = None
    CREDIT_INCOME_RATIO: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    AMT_CREDIT: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    AMT_INCOME_TOTAL: Optional[float] = None
    INCOME_PER_PERSON: Optional[float] = None
    DAYS_EMPLOYED: Optional[float] = None
    DAYS_EMPLOYED_PERCENT: Optional[float] = None
    DAYS_BIRTH: Optional[float] = None
    DAYS_ID_PUBLISH: Optional[float] = None
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    DAYS_REGISTRATION: Optional[float] = None
    AGE_YEARS: Optional[float] = None
    OWN_CAR_AGE: Optional[float] = None
    REGION_POPULATION_RELATIVE: Optional[float] = None
    TOTALAREA_MODE: Optional[float] = None
    BASEMENTAREA_MODE: Optional[float] = None
    b_avg_utilization: Optional[float] = None
    b_avg_loan_duration: Optional[float] = None
    b_total_current_debt: Optional[float] = None
    b_total_history_months: Optional[float] = None
    b_total_loan_count: Optional[float] = None
    b_active_loan_count: Optional[float] = None
    b_closed_loan_count: Optional[float] = None
    int_max_ins_days_late_ever: Optional[float] = None
    int_total_remaining_installments: Optional[float] = None
    int_avg_payment_performance: Optional[float] = None
    int_total_remaining_debt: Optional[float] = None
    int_total_prev_loans_count: Optional[float] = None
    int_max_pos_dpd_ever: Optional[float] = None
    cc_total_avg_utilization_ratio: Optional[float] = None
    cc_total_credit_card_experience_months: Optional[float] = None
    cc_avg_repayment_performance: Optional[float] = None
    cc_total_transaction_count: Optional[float] = None
    cc_max_balance_ever: Optional[float] = None
    cc_total_current_debt: Optional[float] = None
    ORGANIZATION_TYPE_ENC: Optional[float] = None
    NAME_EDUCATION_TYPE_ENC: Optional[float] = None
    OCCUPATION_TYPE_ENC: Optional[float] = None
    CODE_GENDER_ENC: Optional[float] = None
    WEEKDAY_APPR_PROCESS_START_ENC: Optional[float] = None
    HOUR_APPR_PROCESS_START: Optional[float] = None

    model_config = {"extra": "ignore"}


class FeatureContribution(BaseModel):
    rank: int
    feature: str
    gain_norm: float
    direction: str  # "risk_increasing" | "risk_reducing" | "neutral"


class ExplainResponse(BaseModel):
    explanation_type: str  # "importance_fallback" (SHAP not available)
    model_used: str
    proba_lgbm: float
    top_features: list[FeatureContribution]
    warnings: list[str]


class FeatureEntry(BaseModel):
    rank: int
    name: str
    gain_norm: float
    split_norm: float


class FeaturesResponse(BaseModel):
    count: int
    features: list[FeatureEntry]


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    missing_models: list[str]
    version: str
    auc_holdout: dict[str, float]
    thresholds: dict[str, float]
