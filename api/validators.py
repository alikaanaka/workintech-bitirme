"""Business rule validation — field-level and cross-field (model-level)."""

from __future__ import annotations

from api.schemas import FieldValidationError, PredictRequest


def validate_field_rules(req: PredictRequest) -> list[FieldValidationError]:
    errors: list[FieldValidationError] = []

    if req.AMT_INCOME_TOTAL is not None and req.AMT_INCOME_TOTAL <= 0:
        errors.append(FieldValidationError(
            field="AMT_INCOME_TOTAL",
            message="Must be greater than 0.",
        ))

    for field in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        val = getattr(req, field)
        if val is not None and not (0.0 <= val <= 1.0):
            errors.append(FieldValidationError(
                field=field,
                message="Must be in [0, 1].",
            ))

    if req.CODE_GENDER_ENC is not None and req.CODE_GENDER_ENC not in (0, 1, 0.0, 1.0):
        errors.append(FieldValidationError(
            field="CODE_GENDER_ENC",
            message="Must be 0 (F) or 1 (M).",
        ))

    for field in ("DAYS_EMPLOYED", "DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"):
        val = getattr(req, field)
        if val is not None and val > 0:
            errors.append(FieldValidationError(
                field=field,
                message="Must be <= 0 (days relative to application date).",
            ))

    return errors


def validate_cross_field_rules(req: PredictRequest) -> list[FieldValidationError]:
    errors: list[FieldValidationError] = []

    if req.AMT_ANNUITY is not None and req.AMT_CREDIT is not None and req.AMT_CREDIT > 0:
        if req.AMT_ANNUITY > req.AMT_CREDIT / 6:
            errors.append(FieldValidationError(
                field="AMT_ANNUITY",
                message=f"Annuity ({req.AMT_ANNUITY}) exceeds AMT_CREDIT/6 ({req.AMT_CREDIT / 6:.2f}). Unlikely loan structure.",
            ))
        if req.AMT_ANNUITY < req.AMT_CREDIT / 360:
            errors.append(FieldValidationError(
                field="AMT_ANNUITY",
                message=f"Annuity ({req.AMT_ANNUITY}) is below AMT_CREDIT/360 ({req.AMT_CREDIT / 360:.2f}). Unlikely loan structure.",
            ))

    if req.AMT_GOODS_PRICE is not None and req.AMT_CREDIT is not None and req.AMT_CREDIT > 0:
        if req.AMT_GOODS_PRICE > req.AMT_CREDIT * 1.2:
            errors.append(FieldValidationError(
                field="AMT_GOODS_PRICE",
                message=f"Goods price ({req.AMT_GOODS_PRICE}) exceeds AMT_CREDIT × 1.2 ({req.AMT_CREDIT * 1.2:.2f}).",
            ))

    return errors


def run_all_validations(req: PredictRequest) -> tuple[list[FieldValidationError], list[FieldValidationError]]:
    """Return (field_errors, cross_field_errors)."""
    return validate_field_rules(req), validate_cross_field_rules(req)
