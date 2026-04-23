# Aktif Model Özeti — Home Credit Default Risk

**Tarih:** 2026-04-22  
**Görev:** Kredi temerrüt riski tahmini (binary classification, ~8% pozitif oran)  
**Veri:** `train_merged.csv` (~615K satır) + `previous_application.csv` (sıralı geçmiş)

---

## Aktif Modeller

### 1. LightGBM Top-50 CV5

| Parametre | Değer |
|---|---|
| **Dosya** | `models_saved/lgbm_top50_cv5_threshold030_20260421.pkl` |
| **Script** | `src/training/train_lgbm_top50_cv5.py` |
| Feature sayısı | 50 (gain importance sıralaması) |
| scale_pos_weight | 11.39 (neg/pos oranı, her fold'da otomatik) |
| Strateji | 5-fold stratified CV + final model |
| Threshold | 0.30 |

| Metrik | CV Ortalama ± Std | Holdout |
|---|---:|---:|
| AUC-ROC | 0.8800 ± 0.0020 | 0.8783 |
| Gini | 0.7601 ± 0.0039 | 0.7566 |
| KS | 0.6229 ± 0.0029 | 0.6219 |
| PR-AUC | 0.4374 ± 0.0030 | 0.4331 |
| F1 @ 0.30 | 0.3103 ± 0.0014 | 0.3111 |
| Best F1 threshold | — | 0.60 → F1=0.470 |

---

### 2. LSTM OOF CV5

| Parametre | Değer |
|---|---|
| **Dosya** | `models_saved/lstm_oof_cv5_20260422.pt` |
| **Script** | `src/training/train_lstm.py` |
| Mimari | HybridLSTMClassifier — BiLSTM (hidden=128, 2 katman) + ScaledDotAttention + LayerNorm |
| Strateji | 5-fold OOF cross-validation |
| Loss | FocalLoss (γ=2.0, pos_weight=neg/pos) |
| SEQUENCE_MAX_LEN | 15 · batch=512 · AdamW (lr=1e-3, wd=1e-4) · grad_clip=1.0 |

| Metrik | OOF (492K satır) | Holdout |
|---|---:|---:|
| AUC | 0.8733 (overall) | 0.8914 |
| Gini | — | 0.7827 |
| KS | — | 0.6481 |
| PR-AUC | — | 0.5876 |
| F1 @ optimal threshold | 0.5310 @ 0.612 | 0.5481 @ 0.6086 |

---

### 3. Ensemble — LightGBM Top-50 + LSTM OOF CV5 ★ EN İYİ

| Parametre | Değer |
|---|---|
| **Dosya** | `models_saved/ensemble_top50_lstm_20260422.pkl` |
| **Script** | `src/training/train_ensemble_top50_lstm.py` |
| Yöntem | Score blending: `0.55 × lgbm_score + 0.45 × lstm_score` |
| Ağırlık seçimi | OOF F1 grid search (0→1, step=0.05) |
| Threshold | 0.6322 (OOF F1 optimize edildi) |

| Model | Holdout AUC | Gini | KS | PR-AUC | F1 @ 0.6322 |
|---|---:|---:|---:|---:|---:|
| LGBM Top-50 standalone | 0.8783 | 0.7566 | 0.6219 | 0.4331 | 0.4715 |
| LSTM OOF CV5 standalone | 0.8914 | 0.7827 | 0.6481 | 0.5876 | 0.5463 |
| **Ensemble** | **0.9114** | **0.8229** | **0.6889** | **0.6100** | **0.5704** |

**Confusion Matrix (holdout):** `TP=5800, FP=4606, TN=108469, FN=4130`

---

## Artifact Haritası

```
models_saved/
├── lgbm_top50_cv5_threshold030_20260421.pkl   ← LGBM Top-50
├── lstm_oof_cv5_20260422.pt                   ← LSTM final model
├── lstm_oof_predictions_20260422.npz          ← OOF tahminleri (stacking için hazır)
├── lstm_holdout_predictions_20260422.npz      ← Holdout tahminleri
└── ensemble_top50_lstm_20260422.pkl           ← Ensemble bundle ★

data/artifacts/feature_lists/
├── top50_features_lgbm_train_feature_cv5.json          ← 50 feature listesi (kaynak)
├── feature_importance_lgbm_top50_cv5.json              ← LGBM gain/split önem değerleri
└── feature_importance_ensemble_top50_lstm.json         ← Ensemble LGBM+LSTM ağırlıkları

reports/
├── metrics_lgbm_top50_cv5_20260421.md
├── lstm_oof_cv5_report_20260422.md
├── ensemble_top50_lstm_report_20260422.md
├── active_model_summary_20260422.md           ← bu dosya
└── lgbm_top50_cv5_pr_threshold_chart_*.png
```

---

## Feature Önem Ağırlıkları

### LGBM Top-50 — Feature Önemi (Gain)

> **Kaynak:** `data/artifacts/feature_lists/feature_importance_lgbm_top50_cv5.json`  
> `gain_normalized`: tüm ağacın toplam kazancındaki pay  

| Sıra | Feature | Gain Norm | Split Norm |
|---:|---|---:|---:|
| 1 | EXT_SOURCE_MEAN | 0.1600 | 0.0264 |
| 2 | CREDIT_TERM | 0.0447 | 0.0360 |
| 3 | EXT_SOURCE_3 | 0.0301 | 0.0334 |
| 4 | b_avg_utilization | 0.0288 | 0.0270 |
| 5 | EXT_SOURCE_1 | 0.0271 | 0.0274 |
| 6 | EXT_SOURCE_2 | 0.0264 | 0.0305 |
| 7 | EXT_SOURCE_PROD | 0.0259 | 0.0244 |
| 8 | DAYS_EMPLOYED_PERCENT | 0.0256 | 0.0306 |
| 9 | DAYS_EMPLOYED | 0.0248 | 0.0275 |
| 10 | int_max_ins_days_late_ever | 0.0243 | 0.0272 |
| 11 | b_avg_loan_duration | 0.0242 | 0.0342 |
| 12 | EXT_SOURCE_STD | 0.0239 | 0.0348 |
| 13 | DAYS_ID_PUBLISH | 0.0231 | 0.0304 |
| 14 | DAYS_LAST_PHONE_CHANGE | 0.0227 | 0.0295 |
| 15 | DAYS_REGISTRATION | 0.0224 | 0.0318 |
| 16 | AMT_ANNUITY | 0.0221 | 0.0258 |
| 17 | b_total_current_debt | 0.0213 | 0.0271 |
| 18 | int_total_remaining_installments | 0.0205 | 0.0202 |
| 19 | int_avg_payment_performance | 0.0204 | 0.0212 |
| 20 | ANNUITY_INCOME_RATIO | 0.0203 | 0.0257 |
| 21 | TOTALAREA_MODE | 0.0201 | 0.0257 |
| 22 | AMT_CREDIT | 0.0186 | 0.0202 |
| 23 | AMT_GOODS_PRICE | 0.0181 | 0.0186 |
| 24 | REGION_POPULATION_RELATIVE | 0.0176 | 0.0246 |
| 25 | AGE_YEARS | 0.0175 | 0.0213 |
| 26 | CREDIT_INCOME_RATIO | 0.0171 | 0.0228 |
| 27 | int_total_remaining_debt | 0.0165 | 0.0192 |
| 28 | OWN_CAR_AGE | 0.0163 | 0.0166 |
| 29 | DAYS_BIRTH | 0.0156 | 0.0165 |
| 30 | BASEMENTAREA_MODE | 0.0154 | 0.0212 |
| 31 | INCOME_PER_PERSON | 0.0146 | 0.0201 |
| 32 | b_total_history_months | 0.0137 | 0.0188 |
| 33 | int_total_prev_loans_count | 0.0131 | 0.0120 |
| 34 | cc_total_avg_utilization_ratio | 0.0128 | 0.0102 |
| 35 | AMT_INCOME_TOTAL | 0.0113 | 0.0161 |
| 36 | cc_total_credit_card_experience_months | 0.0111 | 0.0125 |
| 37 | b_total_loan_count | 0.0101 | 0.0123 |
| 38 | ORGANIZATION_TYPE_ENC | 0.0093 | 0.0136 |
| 39 | NAME_EDUCATION_TYPE_ENC | 0.0090 | 0.0041 |
| 40 | b_active_loan_count | 0.0089 | 0.0091 |
| 41 | OCCUPATION_TYPE_ENC | 0.0089 | 0.0120 |
| 42 | HOUR_APPR_PROCESS_START | 0.0088 | 0.0136 |
| 43 | cc_avg_repayment_performance | 0.0084 | 0.0106 |
| 44 | b_closed_loan_count | 0.0082 | 0.0111 |
| 45 | CODE_GENDER_ENC | 0.0073 | 0.0034 |
| 46 | cc_total_transaction_count | 0.0072 | 0.0092 |
| 47 | cc_max_balance_ever | 0.0070 | 0.0088 |
| 48 | cc_total_current_debt | 0.0067 | 0.0074 |
| 49 | WEEKDAY_APPR_PROCESS_START_ENC | 0.0065 | 0.0098 |
| 50 | int_max_pos_dpd_ever | 0.0056 | 0.0077 |

---

### Ensemble — Efektif Feature Ağırlıkları

> **Kaynak:** `data/artifacts/feature_lists/feature_importance_ensemble_top50_lstm.json`  
> `effective_gain = gain_normalized × 0.55` (LGBM'nin ensemble'daki ağırlıklı katkısı)  
> LSTM bileşeninin 7 sequence feature'ı model çıktısına `× 0.45` ağırlıkla katkıda bulunur — tek feature bazında gain hesaplanamaz.

#### LGBM Bileşeni (Top-20, tam liste JSON'da)

| Sıra | Feature | Gain Norm | Efektif Gain (×0.55) |
|---:|---|---:|---:|
| 1 | EXT_SOURCE_MEAN | 0.1600 | 0.0880 |
| 2 | CREDIT_TERM | 0.0447 | 0.0246 |
| 3 | EXT_SOURCE_3 | 0.0301 | 0.0165 |
| 4 | b_avg_utilization | 0.0288 | 0.0159 |
| 5 | EXT_SOURCE_1 | 0.0271 | 0.0149 |
| 6 | EXT_SOURCE_2 | 0.0264 | 0.0145 |
| 7 | EXT_SOURCE_PROD | 0.0259 | 0.0143 |
| 8 | DAYS_EMPLOYED_PERCENT | 0.0256 | 0.0141 |
| 9 | DAYS_EMPLOYED | 0.0248 | 0.0137 |
| 10 | int_max_ins_days_late_ever | 0.0243 | 0.0134 |
| 11 | b_avg_loan_duration | 0.0242 | 0.0133 |
| 12 | EXT_SOURCE_STD | 0.0239 | 0.0132 |
| 13 | DAYS_ID_PUBLISH | 0.0231 | 0.0127 |
| 14 | DAYS_LAST_PHONE_CHANGE | 0.0227 | 0.0125 |
| 15 | DAYS_REGISTRATION | 0.0224 | 0.0123 |
| 16 | AMT_ANNUITY | 0.0221 | 0.0122 |
| 17 | b_total_current_debt | 0.0213 | 0.0117 |
| 18 | int_total_remaining_installments | 0.0205 | 0.0113 |
| 19 | int_avg_payment_performance | 0.0204 | 0.0112 |
| 20 | ANNUITY_INCOME_RATIO | 0.0203 | 0.0112 |

#### LSTM Bileşeni — Sequence Feature'lar

| Feature | Kaynak | Ensemble Katkısı |
|---|---|---:|
| AMT_CREDIT | previous_application (sıralı) | × 0.45 (model çıktısı) |
| AMT_APPLICATION | previous_application (sıralı) | × 0.45 (model çıktısı) |
| AMT_ANNUITY | previous_application (sıralı) | × 0.45 (model çıktısı) |
| CNT_PAYMENT | previous_application (sıralı) | × 0.45 (model çıktısı) |
| DAYS_DECISION | previous_application (sıralı) | × 0.45 (model çıktısı) |
| NAME_CONTRACT_TYPE | previous_application (sıralı) | × 0.45 (model çıktısı) |
| NAME_CONTRACT_STATUS | previous_application (sıralı) | × 0.45 (model çıktısı) |

> Not: LSTM, geçmişi olmayan müşterilerde mask tamamen sıfır olur → ScaledDotAttention uniform attention uygular → sadece static feature'larla tahmin üretir.

---

## Ensemble Formülü

```
LGBM Top-50  →  P(default) ─┐
                              ├→  0.55 × lgbm + 0.45 × lstm  →  final skor  →  threshold=0.6322  →  binary
LSTM OOF CV5 →  P(default) ─┘
```

**Ağırlık seçimi:** OOF F1 grid search (α ∈ [0.00, 1.00], step=0.05) → α=0.55  
**Threshold seçimi:** OOF üzerinde F1 maximizasyonu → 0.6322
