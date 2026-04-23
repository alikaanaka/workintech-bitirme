"""Model artifact loader and inference engine."""

from __future__ import annotations

import os
import pickle
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

# Both OMP env vars must be set before any library loads its runtime.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Import both OpenMP-linked libraries at module level so their runtimes
# are initialized together — importing one then the other at runtime causes segfault.
import lightgbm  # noqa: F401
import torch
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # already started — harmless

import numpy as np
import pandas as pd

from api import config as api_cfg

logger = logging.getLogger(__name__)

# Top-50 feature names (post-encoding) — must match bundle["feature_columns"]
_TOP50_FEATURES = [
    "EXT_SOURCE_MEAN", "CREDIT_TERM", "EXT_SOURCE_3", "b_avg_utilization",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_PROD", "DAYS_EMPLOYED_PERCENT",
    "DAYS_EMPLOYED", "int_max_ins_days_late_ever", "b_avg_loan_duration",
    "EXT_SOURCE_STD", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE",
    "DAYS_REGISTRATION", "AMT_ANNUITY", "b_total_current_debt",
    "int_total_remaining_installments", "int_avg_payment_performance",
    "ANNUITY_INCOME_RATIO", "TOTALAREA_MODE", "AMT_CREDIT", "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE", "AGE_YEARS", "CREDIT_INCOME_RATIO",
    "int_total_remaining_debt", "OWN_CAR_AGE", "DAYS_BIRTH",
    "BASEMENTAREA_MODE", "INCOME_PER_PERSON", "b_total_history_months",
    "int_total_prev_loans_count", "cc_total_avg_utilization_ratio",
    "AMT_INCOME_TOTAL", "cc_total_credit_card_experience_months",
    "b_total_loan_count", "ORGANIZATION_TYPE_ENC", "NAME_EDUCATION_TYPE_ENC",
    "b_active_loan_count", "OCCUPATION_TYPE_ENC", "HOUR_APPR_PROCESS_START",
    "cc_avg_repayment_performance", "b_closed_loan_count", "CODE_GENDER_ENC",
    "cc_total_transaction_count", "cc_max_balance_ever",
    "cc_total_current_debt", "WEEKDAY_APPR_PROCESS_START_ENC",
    "int_max_pos_dpd_ever",
]


@dataclass
class _LGBMBundle:
    model: Any
    feature_columns: list[str]


@dataclass
class _LSTMBundle:
    model: Any  # HybridLSTMClassifier
    static_scaler: Any  # StandardScaler
    seq_scaler: Any  # StandardScaler
    encoder: Any  # FoldEncoder
    categorical_maps: dict
    static_dim: int
    sequence_dim: int
    full_feature_names: list[str]  # all encoded column names in scaler order


@dataclass
class _EnsembleBundle:
    lgbm_weight: float
    lstm_weight: float
    threshold: float


@dataclass
class InferenceResult:
    proba_lgbm: float
    proba_lstm: Optional[float]
    proba_ensemble: Optional[float]
    available_models: list[str]
    threshold_used: float
    warnings: list[str] = field(default_factory=list)


class ModelPredictor:
    """Loads artifacts at startup; provides inference methods."""

    def __init__(self) -> None:
        self._lgbm: Optional[_LGBMBundle] = None
        self._lstm: Optional[_LSTMBundle] = None
        self._ensemble: Optional[_EnsembleBundle] = None

    # ------------------------------------------------------------------
    # Startup loading
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        self._lgbm = self._load_lgbm()
        self._lstm = self._load_lstm()
        self._ensemble = self._load_ensemble()

    def _load_lgbm(self) -> Optional[_LGBMBundle]:
        path = api_cfg.MODELS_SAVED_DIR / api_cfg.LGBM_FILE
        if not path.exists():
            logger.error("LGBM bundle not found: %s", path)
            raise FileNotFoundError(f"Required LGBM model not found: {path}")
        try:
            with open(path, "rb") as fh:
                bundle = pickle.load(fh)
            feature_cols = bundle.get("feature_columns", _TOP50_FEATURES)
            logger.info("LGBM loaded — %d features", len(feature_cols))
            return _LGBMBundle(model=bundle["model"], feature_columns=feature_cols)
        except Exception as exc:
            logger.error("Failed to load LGBM bundle: %s", exc)
            raise

    def _load_lstm(self) -> Optional[_LSTMBundle]:
        from src.models.lstm_model import HybridLSTMClassifier

        path = api_cfg.MODELS_SAVED_DIR / api_cfg.LSTM_FILE
        if not path.exists():
            logger.warning("LSTM bundle not found: %s — LSTM disabled", path)
            return None
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            static_dim: int = checkpoint["static_dim"]
            sequence_dim: int = checkpoint["sequence_dim"]
            encoder = checkpoint["encoder"]
            static_scaler = checkpoint["static_scaler"]
            seq_scaler = checkpoint["seq_scaler"]
            categorical_maps = checkpoint["categorical_maps"]

            # Derive encoded feature names in scaler order via a dummy transform
            dummy_num = {col: 0.0 for col in encoder._numerical_columns}
            dummy_cat = {col: encoder._category_maps[col].get("MISSING", list(encoder._category_maps[col].values())[0])
                         for col in encoder._categorical_columns}
            dummy_df = pd.DataFrame([{**dummy_num, **dummy_cat}])
            encoded_dummy = encoder.transform(dummy_df)
            full_feature_names = list(encoded_dummy.columns)

            model = HybridLSTMClassifier(
                static_dim=static_dim,
                sequence_dim=sequence_dim,
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()

            logger.info("LSTM loaded — static_dim=%d sequence_dim=%d", static_dim, sequence_dim)
            return _LSTMBundle(
                model=model,
                static_scaler=static_scaler,
                seq_scaler=seq_scaler,
                encoder=encoder,
                categorical_maps=categorical_maps,
                static_dim=static_dim,
                sequence_dim=sequence_dim,
                full_feature_names=full_feature_names,
            )
        except Exception as exc:
            logger.warning("Failed to load LSTM bundle: %s — LSTM disabled", exc)
            return None

    def _load_ensemble(self) -> Optional[_EnsembleBundle]:
        path = api_cfg.MODELS_SAVED_DIR / api_cfg.ENSEMBLE_FILE
        if not path.exists():
            logger.warning("Ensemble bundle not found: %s — ensemble disabled", path)
            return None
        try:
            with open(path, "rb") as fh:
                bundle = pickle.load(fh)
            lgbm_w = bundle.get("lgbm_weight", api_cfg.ENSEMBLE_WEIGHT_LGBM)
            lstm_w = bundle.get("lstm_weight", api_cfg.ENSEMBLE_WEIGHT_LSTM)
            threshold = bundle.get("threshold", api_cfg.THRESHOLD_ENSEMBLE)
            logger.info("Ensemble loaded — lgbm_weight=%.2f lstm_weight=%.2f threshold=%.4f",
                        lgbm_w, lstm_w, threshold)
            return _EnsembleBundle(lgbm_weight=lgbm_w, lstm_weight=lstm_w, threshold=threshold)
        except Exception as exc:
            logger.warning("Failed to load ensemble bundle: %s — ensemble disabled", exc)
            return None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def available_model_names(self) -> list[str]:
        names = []
        if self._lgbm is not None:
            names.append("lgbm")
        if self._lstm is not None:
            names.append("lstm")
        if self._ensemble is not None and self._lgbm is not None and self._lstm is not None:
            names.append("ensemble")
        return names

    @property
    def missing_model_names(self) -> list[str]:
        available = set(self.available_model_names)
        return [m for m in ("lgbm", "lstm", "ensemble") if m not in available]

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _score_lgbm(self, features: dict) -> float:
        assert self._lgbm is not None
        row = {col: (v if v is not None else np.nan) for col, v in
               ((c, features.get(c)) for c in self._lgbm.feature_columns)}
        df = pd.DataFrame([row], dtype=np.float64)
        return float(self._lgbm.model.predict_proba(df)[:, 1][0])

    def _score_lstm(self, features: dict, prev_apps: list[dict]) -> Optional[float]:
        if self._lstm is None:
            return None
        from src.models.lstm_model import build_sequence_dataset, SEQUENCE_MAX_LEN
        from src.training.train_lstm import _apply_sequence_scaler

        bundle = self._lstm

        # --- static features ---
        # Build a full-width DataFrame (all encoded columns, unknown → 0.0 / NaN for numerics)
        # First fill categorical columns with MISSING token so encoder can map them
        num_vals = {col: features.get(col, np.nan) for col in bundle.encoder._numerical_columns}
        cat_vals = {}
        for col in bundle.encoder._categorical_columns:
            cat_vals[col] = "MISSING"
        raw_df = pd.DataFrame([{**num_vals, **cat_vals}])

        # Apply encoder → all encoded features in correct order
        try:
            enc_df = bundle.encoder.transform(raw_df).fillna(0.0)
        except Exception:
            return None

        # Align to the full feature order the scaler expects
        full_vec = np.zeros((1, bundle.static_dim), dtype=np.float32)
        for i, fname in enumerate(bundle.full_feature_names[:bundle.static_dim]):
            if fname in enc_df.columns:
                full_vec[0, i] = float(enc_df[fname].iloc[0])
            # Provide user-supplied top-50 values where they map directly
            elif fname in features:
                full_vec[0, i] = float(features[fname]) if features[fname] is not None else 0.0

        static_scaled = bundle.static_scaler.transform(full_vec).astype(np.float32)

        # --- sequence features ---
        dummy_customer_id = 0
        if prev_apps:
            prev_records = []
            for app in prev_apps:
                record = {**app, "SK_ID_CURR": dummy_customer_id}
                prev_records.append(record)
            prev_df = pd.DataFrame(prev_records)
        else:
            prev_df = pd.DataFrame(columns=["SK_ID_CURR"])

        from src.config import config as src_cfg
        orig_key = src_cfg.KEY_COLUMN
        prev_df_keyed = prev_df.rename(columns={"SK_ID_CURR": orig_key}) if "SK_ID_CURR" in prev_df.columns else prev_df

        seq_art = build_sequence_dataset(
            prev_df_keyed,
            customer_ids=np.array([dummy_customer_id]),
            max_len=SEQUENCE_MAX_LEN,
            categorical_maps=bundle.categorical_maps,
        )

        seq_scaled = _apply_sequence_scaler(seq_art.sequences, seq_art.masks, bundle.seq_scaler)

        # --- model forward ---
        static_t = torch.tensor(static_scaled, dtype=torch.float32)
        seq_t = torch.tensor(seq_scaled, dtype=torch.float32)
        mask_t = torch.tensor(seq_art.masks, dtype=torch.float32)

        with torch.no_grad():
            logit = bundle.model(static_t, seq_t, mask_t)
            proba = float(torch.sigmoid(logit).item())

        return proba

    # ------------------------------------------------------------------
    # Public inference
    # ------------------------------------------------------------------

    def predict(self, features: dict, prev_apps: Optional[list[dict]] = None) -> InferenceResult:
        """
        features: dict of {feature_name: value} for the top-50 LGBM features
        prev_apps: optional list of previous application dicts for LSTM
        """
        if self._lgbm is None:
            raise RuntimeError("LGBM model not loaded — cannot predict.")

        warnings: list[str] = []
        has_sequence = bool(prev_apps)

        proba_lgbm = self._score_lgbm(features)
        proba_lstm: Optional[float] = None
        proba_ensemble: Optional[float] = None
        available: list[str] = ["lgbm"]
        threshold_used = api_cfg.THRESHOLD_LGBM

        if has_sequence and self._lstm is not None:
            try:
                proba_lstm = self._score_lstm(features, prev_apps or [])
                available.append("lstm")
            except Exception as exc:
                logger.warning("LSTM inference failed: %s", exc)
                warnings.append("LSTM inference failed; using LGBM only.")
        elif has_sequence and self._lstm is None:
            warnings.append("LSTM model not loaded; sequence data ignored.")

        if proba_lstm is not None and self._ensemble is not None:
            ens = self._ensemble
            proba_ensemble = ens.lgbm_weight * proba_lgbm + ens.lstm_weight * proba_lstm
            available.append("ensemble")
            threshold_used = ens.threshold
        elif proba_lstm is None and self._ensemble is not None and not has_sequence:
            # Manual mode: ensemble not applicable without sequence
            pass

        return InferenceResult(
            proba_lgbm=proba_lgbm,
            proba_lstm=proba_lstm,
            proba_ensemble=proba_ensemble,
            available_models=available,
            threshold_used=threshold_used,
            warnings=warnings,
        )


# Module-level singleton — shared across requests
predictor = ModelPredictor()
