# Proje Durum Özeti — Home Credit Default Risk

**Tarih:** 2026-04-22  
**Görev:** Kredi temerrüt riski tahmini (binary classification, ~8% pozitif oran)  
**Veri:** `train_merged.csv` (~615K satır) + `previous_application.csv` (sıralı geçmiş)

---

## Model Gelişim Kronolojisi

### Phase 1 — Baseline LightGBM (lgbm_model.pkl)
İlk model, tüm featurelar, sabit threshold.

### Phase 2 — LightGBM v2 (lgbm_v2 + lgbm_train_feature_cv5)
5-fold CV, FoldEncoder (target encoding), scale_pos_weight, erken durdurma.

| Model | Holdout AUC | F1 @ 0.30 |
|---|---:|---:|
| LightGBM v2 | 0.893654 | 0.336 |
| LightGBM train_feature CV5 | 0.893654 | 0.281 |
| CatBoost CV5 | 0.803056 | 0.291 |
| XGBoost CV5 | — | — |

### Phase 3 v1 — LSTM (Başarısız)
Sequence featurelar normalize edilmeden LSTM eklendi. Model neredeyse herkesi pozitif tahmin etti.

| Model | Holdout AUC | F1 @ 0.30 |
|---|---:|---:|
| LSTM v1 | ~0.531 | 0.149 |
| Ensemble v1 (LGBM+LSTM) | ~0.934 | 0.149 |

**Sorun:** AMT_CREDIT (milyonlar) ve DAYS_DECISION (-2000) normalize edilmemişti → LSTM AUC ≈ 0.50 (rastgele).

### Phase 3 v2 — LSTM İyileştirildi (lstm_model.pt — eski)
Sequence + static StandardScaler, ScaledDotAttention, LayerNorm, FocalLoss(γ=2.0), gradient clipping, F1-optimal threshold.

| Model | Holdout AUC | F1 | Threshold |
|---|---:|---:|---:|
| LSTM v2 | 0.869516 | 0.525940 | 0.6020 |
| Ensemble v2 (LGBM+LSTM) | 0.895893 | 0.579880 | 0.6020 |

---

## Güncel Modeller (Aktif)

### 1. LightGBM Top-50 CV5
**Dosya:** `models_saved/lgbm_top50_cv5_threshold030_20260421.pkl`  
**Script:** `src/training/train_lgbm_top50_cv5.py`

- **Feature sayısı:** 50 (önceki CV5 modelinin gain importance top-50'si)
- **scale_pos_weight:** 11.39 (neg/pos oranı, her fold'da otomatik)
- **Strateji:** 5-fold stratified CV + final model

| Metrik | CV Ortalama ± Std | Holdout |
|---|---:|---:|
| AUC-ROC | 0.8800 ± 0.0020 | 0.8783 |
| Gini | 0.7601 ± 0.0039 | 0.7566 |
| KS | 0.6229 ± 0.0029 | 0.6219 |
| PR-AUC | 0.4374 ± 0.0030 | 0.4331 |
| F1 @ 0.30 | 0.3103 ± 0.0014 | 0.3111 |
| Best F1 threshold | — | 0.60 → F1=0.470 |

**Top-5 feature:** EXT_SOURCE_MEAN, CREDIT_TERM, EXT_SOURCE_PROD, EXT_SOURCE_3, b_avg_utilization

---

### 2. LSTM OOF CV5
**Dosya:** `models_saved/lstm_oof_cv5_20260422.pt`  
**Script:** `src/training/train_lstm.py` → `train_lstm_oof_cv5()`

- **Mimari:** HybridLSTMClassifier — BiLSTM (hidden=128, 2 katman) + ScaledDotAttention + LayerNorm
- **Strateji:** 5-fold OOF cross-validation (her fold kendi encoder/scaler/categorical_maps ile)
- **Loss:** FocalLoss (γ=2.0, pos_weight=neg/pos)
- **SEQUENCE_MAX_LEN:** 15 | batch=512 | AdamW (lr=1e-3, wd=1e-4) | grad_clip=1.0
- **OOF tahminleri:** `models_saved/lstm_oof_predictions_20260422.npz` (stacking için hazır)

| Metrik | OOF (492K satır) | Holdout |
|---|---:|---:|
| AUC | 0.8733 (overall) | 0.8914 |
| Gini | — | 0.7827 |
| KS | — | 0.6481 |
| PR-AUC | — | 0.5876 |
| F1 @ optimal threshold | 0.5310 @ 0.612 | 0.5481 @ 0.6086 |

**Fold bazlı OOF AUC:**

| Fold | OOF AUC | Best Epoch |
|---:|---:|---:|
| 1 | 0.8740 | 50 |
| 2 | 0.8751 | 49 |
| 3 | 0.8747 | 50 |
| 4 | 0.8723 | 48 |
| 5 | 0.8712 | 49 |
| **Ort ± Std** | **0.8735 ± 0.0015** | — |

---

### 3. Ensemble — LightGBM Top-50 + LSTM OOF CV5 ★ EN İYİ
**Dosya:** `models_saved/ensemble_top50_lstm_20260422.pkl`  
**Script:** `src/training/train_ensemble_top50_lstm.py`

- **Yöntem:** Score blending — `0.55 × lgbm_score + 0.45 × lstm_score`
- **Ağırlık seçimi:** OOF F1 grid search (leak-free, train_full üzerinde)
- **Threshold:** 0.6322 (OOF F1 optimize edildi)

| Model | Holdout AUC | Gini | KS | PR-AUC | F1 @ 0.6322 |
|---|---:|---:|---:|---:|---:|
| LGBM Top-50 standalone | 0.8783 | 0.7566 | 0.6219 | 0.4331 | 0.4715 |
| LSTM OOF CV5 standalone | 0.8914 | 0.7827 | 0.6481 | 0.5876 | 0.5463 |
| **Ensemble** | **0.9114** | **0.8229** | **0.6889** | **0.6100** | **0.5704** |

**Confusion Matrix (holdout):**  
`TP=5800, FP=4606, TN=108469, FN=4130`

---

## Tüm Modeller Karşılaştırması

| Model | Holdout AUC | F1 | Threshold | Notlar |
|---|---:|---:|---:|---|
| LightGBM v2 (Phase 2) | 0.8937 | 0.2813 | 0.22 | Tam feature, eski |
| LSTM v1 (Phase 3 ilk) | 0.5316 | 0.1494 | 0.30 | Normalize edilmemiş, başarısız |
| Ensemble v1 | ~0.934 | 0.1494 | 0.30 | LSTM kötü olduğu için F1 çöktü |
| LSTM v2 (Phase 3 düzeltilmiş) | 0.8695 | 0.5259 | 0.6020 | Tek fold, eski |
| Ensemble v2 | 0.8959 | 0.5799 | 0.6020 | Eski ensemble, tek fold LSTM |
| **LGBM Top-50 CV5** | **0.8783** | **0.3111** | 0.30 | Top-50 feature, scale_pos_weight |
| **LSTM OOF CV5** | **0.8914** | **0.5481** | 0.6086 | 5-fold OOF, güncel |
| **Ensemble Top50+LSTM OOF** ★ | **0.9114** | **0.5704** | 0.6322 | **En iyi model** |

---

## Artifact Haritası

```
models_saved/
├── lgbm_top50_cv5_threshold030_20260421.pkl   ← LGBM Top-50 (aktif)
├── lstm_oof_cv5_20260422.pt                   ← LSTM final model (aktif)
├── lstm_oof_predictions_20260422.npz          ← OOF tahminleri (stacking için)
├── lstm_holdout_predictions_20260422.npz      ← Holdout tahminleri
└── ensemble_top50_lstm_20260422.pkl           ← Ensemble bundle (aktif) ★

data/artifacts/feature_lists/
├── top50_features_lgbm_train_feature_cv5.json  ← 50 feature (gain importance)
└── top50_features_ensemble_lgbm_lstm.json      ← Ensemble LGBM bileşeni top-50

reports/
├── metrics_lgbm_top50_cv5_20260421.md         ← LGBM Top-50 metrikleri
├── lstm_oof_cv5_report_20260422.md            ← LSTM OOF CV5 metrikleri
├── ensemble_top50_lstm_report_20260422.md     ← Ensemble metrikleri ★
└── lgbm_top50_cv5_pr_threshold_chart_*.png   ← PR eğrisi + threshold analizi
```

---

## Mimari Notlar

### LSTM Geçmişsiz Müşteri Davranışı
`previous_application` kaydı olmayan müşterilerde `sequences` ve `mask` tamamen sıfır.  
`_ScaledDotAttention` all-masked durumda uniform attention uygular → model sadece **static feature**'larla tahmin üretir (0.5 değil, gerçek tahmin).

### Ensemble Yöntemi
Şu an uygulanan **score blending** (ağırlıklı ortalama):
```
LGBM Top-50  →  P(default) ─┐
                              ├→  0.55 × lgbm + 0.45 × lstm  →  final skor
LSTM OOF CV5 →  P(default) ─┘
```
İki model birbirinden **bağımsız** eğitildi. LSTM çıktıları LGBM'e feature olarak girmedi.

### Planlanan: Stacking (Meta-Model)
LSTM OOF tahminleri (`lstm_oof_predictions_20260422.npz`) meta-model için hazır.  
Hedef mimari:
```
Static features   →  LightGBM (OOF skor)  ─┐
                                             ├→  Meta-model (LR / LGBM)
Sequence features →  LSTM (OOF skor)      ─┘
```
**Durum:** Henüz uygulanmadı.

---

## Aktif Scriptler

| Script | Görev |
|---|---|
| `src/training/train_lgbm_top50_cv5.py` | LGBM Top-50, scale_pos_weight, 5-fold CV |
| `src/training/train_lstm.py` | LSTM OOF CV5 eğitimi |
| `src/training/train_ensemble_top50_lstm.py` | Score blending ensemble |
| `src/training/train_lgbm_train_feature_cv.py` | Tam-feature LGBM CV5 (referans) |
