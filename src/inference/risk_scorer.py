"""Probability → risk band + binary decision."""

from __future__ import annotations

from api import config as api_cfg


def score_to_risk_band(risk_score_pct: int) -> str:
    if risk_score_pct <= api_cfg.RISK_BAND_LOW_MAX:
        return "Low"
    if risk_score_pct <= api_cfg.RISK_BAND_MEDIUM_MAX:
        return "Medium"
    return "High"


def score_to_decision(risk_score_pct: int) -> str:
    if risk_score_pct <= api_cfg.DECISION_APPROVE_MAX:
        return "ONAYLA"
    if risk_score_pct <= api_cfg.DECISION_REVIEW_MAX:
        return "INCELE"
    return "REDDET"


def proba_to_risk_score_pct(proba: float) -> int:
    """Normalize probability to 0–100 integer for display."""
    return int(round(min(max(proba * 100, 0), 100)))


def evaluate(proba: float) -> tuple[str, int, str]:
    """Return (risk_band, risk_score_pct, decision) from raw probability."""
    pct = proba_to_risk_score_pct(proba)
    return score_to_risk_band(pct), pct, score_to_decision(pct)
