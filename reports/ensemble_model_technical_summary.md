# Ensemble Model — Teknik Özet
**Proje:** Home Credit Default Risk  
**Model adı:** `ensemble_top50_lstm_20260422`  
**Tarih:** 2026-04-22  
**Görev:** Kredi temerrüt riski tahmini — binary classification (~%8 pozitif oran, ~615K satır)

---

## 1. Veri

| Kaynak | Dosya | Kullanım |
|---|---|---|
| Ana başvuru tablosu | `train_merged.csv` | Statik feature'lar (demografik, finansal, birleşik) |
| Geçmiş başvurular | `previous_application.csv` | LSTM sıralı sequence girdisi |

**Bölümleme:**
- Holdout test: %20 (stratified, `random_state=42`)
- LGBM train_full: %80 → 5-fold CV için
- LSTM train_full: %80 → 5-fold OOF CV için; final model için %10 iç validation ayrıldı

---

## 2. Bileşen Model 1 — LightGBM Top-50 CV5

### Eğitim stratejisi
- **5-fold stratified cross-validation** (StratifiedKFold, shuffle=True, seed=42)
- Her fold'da bağımsız `FoldEncoder` (target encoding) + `scale_pos_weight` hesaplaması
- Final model: tüm train_full verisi üzerinde %20 iç validation ile erken durdurma

### Hiperparametreler

| Parametre | Değer |
|---|---|
| objective | binary |
| metric | auc |
| boosting_type | gbdt |
| learning_rate | 0.05 |
| num_leaves | 64 |
| max_depth | -1 |
| min_child_samples | 100 |
| reg_alpha | 0.1 |
| reg_lambda | 0.1 |
| feature_fraction | 0.8 |
| bagging_fraction | 0.8 |
| bagging_freq | 5 |
| n_estimators (max) | 1000 |
| early_stopping_rounds | 50 |
| **scale_pos_weight** | **11.39** (neg/pos, fold'da otomatik) |

### Feature seçimi
- Kaynak: `train_lgbm_train_feature_cv5` modelinin gain importance top-50'si
- JSON: `data/artifacts/feature_lists/top50_features_lgbm_train_feature_cv5.json`
- **Feature sayısı:** 50 (post-encoding; `_ENC` suffix'li sütunlar dahil)

### Performans

| Metrik | CV Ortalama ± Std | Holdout |
|---|---:|---:|
| AUC-ROC | 0.8800 ± 0.0020 | **0.8783** |
| Gini | 0.7601 ± 0.0039 | 0.7566 |
| KS | 0.6229 ± 0.0029 | 0.6219 |
| PR-AUC | 0.4374 ± 0.0030 | 0.4331 |
| F1 @ 0.30 | 0.3103 ± 0.0014 | 0.3111 |

**Artifact:** `models_saved/lgbm_top50_cv5_threshold030_20260421.pkl`  
**Bundle içeriği:** `model`, `encoder` (FoldEncoder), `feature_columns`, `top50_names`, `threshold`, `scale_pos_weight`

---

## 3. Bileşen Model 2 — LSTM OOF CV5

### Mimari: HybridLSTMClassifier

```
Static features (166 encoded) ──→ Linear(166→128) → LayerNorm → ReLU → Dropout(0.3)
                                                                              │
                                                                         static_feat (128)
                                                                              │
Sequence features (7) ──→ Linear(7→128) → LayerNorm → ReLU              ────┤
  per timestep (max 15)        │                                         combined (384)
                             BiLSTM(128, 2 layers, bidirectional)            │
                               │                                        Linear(384→128) → ReLU → Dropout
                          _ScaledDotAttention                                │
                               │                                        Linear(128→1)
                          pooled (256)                                        │
                                                                          logit → sigmoid → P(default)
```

**Toplam parametre:** ~1.2M

### Sequence feature'lar (7 adet)

| Feature | Tip | Kaynak |
|---|---|---|
| AMT_CREDIT | Numeric | previous_application |
| AMT_APPLICATION | Numeric | previous_application |
| AMT_ANNUITY | Numeric | previous_application |
| CNT_PAYMENT | Numeric | previous_application |
| DAYS_DECISION | Numeric | previous_application |
| NAME_CONTRACT_TYPE | Categorical → integer | previous_application |
| NAME_CONTRACT_STATUS | Categorical → integer | previous_application |

**SEQUENCE_MAX_LEN:** 15 (son 15 başvuru, DAYS_DECISION azalan sıralı)  
**Geçmişsiz müşteri:** mask tamamen sıfır → _ScaledDotAttention uniform attention → sadece static feature'larla tahmin

### Eğitim detayları

| Parametre | Değer |
|---|---|
| Strateji | 5-fold OOF cross-validation |
| Loss | FocalLoss (γ=2.0, pos_weight=neg/pos) |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| max_epochs | 50 |
| early_stopping_patience | 6 |
| gradient_clip_max_norm | 1.0 |
| batch_size | 512 |
| Ön işleme (static) | FoldEncoder → StandardScaler |
| Ön işleme (sequence) | StandardScaler (sadece gerçek pozisyonlara fit) |

### Performans

| Metrik | OOF (Fold ort ± std) | Holdout |
|---|---:|---:|
| AUC | 0.8735 ± 0.0015 | **0.8914** |
| Gini | — | 0.7827 |
| KS | — | 0.6481 |
| PR-AUC | — | 0.5876 |
| F1 @ optimal threshold | 0.5310 @ 0.612 | 0.5481 @ 0.6086 |
| Brier Score | — | 0.1127 |

**Fold bazlı OOF AUC:**

| Fold | OOF AUC | Best Epoch |
|---:|---:|---:|
| 1 | 0.8740 | 50 |
| 2 | 0.8751 | 49 |
| 3 | 0.8747 | 50 |
| 4 | 0.8723 | 48 |
| 5 | 0.8712 | 49 |

**Artifact:** `models_saved/lstm_oof_cv5_20260422.pt`  
**Bundle içeriği:** `state_dict`, `encoder`, `static_scaler`, `seq_scaler`, `categorical_maps`, `static_dim=166`, `sequence_dim=7`

---

## 4. Ensemble — Score Blending

### Yöntem

İki model **birbirinden bağımsız** eğitildi. Ensemble, çıktı olasılıklarının ağırlıklı ortalamasıdır:

```
P_ensemble = α × P_lgbm + (1 - α) × P_lstm
           = 0.55 × P_lgbm + 0.45 × P_lstm
```

Binary karar:
```
karar = 1  eğer  P_ensemble ≥ 0.6322
```

### Ağırlık ve threshold seçimi (OOF, leak-free)

1. **Pass 1 — İlk threshold:** α=0.5 ile OOF blend skorları üzerinde precision-recall eğrisinden F1 maksimum threshold bulundu
2. **Pass 2 — Ağırlık grid search:** α ∈ {0.00, 0.05, …, 1.00}, threshold sabit → OOF F1 maksimize → **α = 0.55**
3. **Pass 3 — Threshold rafine:** α=0.55 ile OOF blend üzerinde tekrar precision-recall → **threshold = 0.6322**

Tüm optimizasyon yalnızca train_full üzerinde yapıldı (holdout hiç görülmedi).

### Positional alignment

LGBM train skorları ve LSTM OOF skorları, aynı `train_test_split(random_state=42)` ile elde edilen `x_train_full`'un satır sırasıyla pozisyonel olarak hizalandı (customer_id merge'e gerek yok; her iki array da 492.017 × 1).

### Performans

| Model | Holdout AUC | Gini | KS | PR-AUC | F1 @ 0.6322 |
|---|---:|---:|---:|---:|---:|
| LGBM Top-50 standalone | 0.8783 | 0.7566 | 0.6219 | 0.4331 | 0.4715 |
| LSTM OOF CV5 standalone | 0.8914 | 0.7827 | 0.6481 | 0.5876 | 0.5463 |
| **Ensemble (α=0.55)** | **0.9114** | **0.8229** | **0.6889** | **0.6100** | **0.5704** |

**AUC artışı:** +3.3 pp LGBM'e, +2.0 pp LSTM'e göre

**Confusion Matrix (holdout, threshold=0.6322):**

|  | Pred: 0 | Pred: 1 |
|---|---:|---:|
| **Gerçek: 0** | 108.469 (TN) | 4.606 (FP) |
| **Gerçek: 1** | 4.130 (FN) | 5.800 (TP) |

- **Precision:** 0.557 | **Recall:** 0.584 | **F1:** 0.570

---

## 5. Feature Önem Ağırlıkları (LGBM bileşeni)

> Tam liste: `data/artifacts/feature_lists/feature_importance_ensemble_top50_lstm.json`  
> `effective_gain = gain_normalized × 0.55` (LGBM'nin ensemble'a katkısı)

| Sıra | Feature | Gain Norm | Efektif Gain |
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

**LSTM bileşeni:** 7 sequence feature, toplam `× 0.45` katkı (model seviyesinde, feature bazında gain hesaplanamaz).

---

## 6. Preprocessing Pipeline

### LGBM için

```
Ham veri
  └─→ clean_dataframe()          # anomali temizleme, tip dönüşümleri
  └─→ add_main_features()        # EXT_SOURCE türevleri, gün/yaş oranları
  └─→ add_interaction_features() # kredi/gelir oranları, banka/taksit özetleri
  └─→ FoldEncoder.fit_transform() # kategorik → integer encode (_ENC)
  └─→ top-50 filter              # gain importance ile seçilmiş 50 sütun
  └─→ LGBMClassifier.predict_proba()
```

### LSTM için

```
Ham veri (statik)
  └─→ clean_dataframe() → add_main_features() → add_interaction_features()
  └─→ FoldEncoder.transform()    # tüm encoded feature'lar (166 sütun)
  └─→ StandardScaler.transform() # statik_scaler
  └─→ HybridLSTMClassifier (static input)

previous_application.csv (sıralı, DAYS_DECISION azalan)
  └─→ build_sequence_dataset()   # padding (max_len=15), categorical_maps
  └─→ StandardScaler.transform() # seq_scaler (sadece gerçek pozisyonlara)
  └─→ HybridLSTMClassifier (sequence input)

sigmoid(logit) → P_lstm
```

### Ensemble

```
P_lgbm  ─┐
           ├→  0.55 × P_lgbm + 0.45 × P_lstm  →  P_ensemble  →  ≥ 0.6322 → TEMERRÜT
P_lstm  ─┘
```

---

## 7. Risk Skoru ve Karar Mantığı (API)

| P_ensemble aralığı | risk_score_pct | risk_band | decision |
|---|---:|---|---|
| 0.00 – 0.30 | 0 – 30 | **Low** | ONAYLA |
| 0.30 – 0.70 | 30 – 70 | **Medium** | INCELE |
| 0.70 – 1.00 | 70 – 100 | **High** | REDDET |

> `risk_score_pct = round(P_ensemble × 100)`  
> Binary karar eşiği (0.6322) risk_band sınırlarından bağımsızdır; ikisi birlikte response'da sunulur.

---

## 8. API Endpoint'leri

| Endpoint | Metod | Açıklama |
|---|---|---|
| `/api/health` | GET | Model durumu, AUC'lar, threshold'lar |
| `/api/predict` | POST | Tahmin — LGBM (sequence yok) veya Ensemble (sequence var) |
| `/api/explain` | POST | Feature contribution (gain-importance fallback, top-10) |
| `/api/features/top50` | GET | 50 feature listesi (rank, gain_norm, split_norm) |

**Inference modu:**
- **Manual:** `previous_applications` dizisi yok → sadece LGBM skoru, threshold=0.30
- **Dataset-backed:** `previous_applications` var → LGBM + LSTM + Ensemble, threshold=0.6322

---

## 9. Artifact Haritası

```
models_saved/
├── lgbm_top50_cv5_threshold030_20260421.pkl      ← LGBM bileşeni
│     keys: model, encoder, feature_columns, top50_names, threshold, scale_pos_weight
├── lstm_oof_cv5_20260422.pt                      ← LSTM bileşeni
│     keys: state_dict, encoder, static_scaler, seq_scaler,
│           categorical_maps, static_dim=166, sequence_dim=7
├── lstm_oof_predictions_20260422.npz             ← OOF tahminleri (stacking için)
│     arrays: customer_ids, oof_scores, y_true  (492.017 satır)
├── lstm_holdout_predictions_20260422.npz         ← Holdout tahminleri
│     arrays: customer_ids, holdout_scores, y_true (123.005 satır)
└── ensemble_top50_lstm_20260422.pkl              ← Ensemble metadata bundle
      keys: lgbm_weight=0.55, lstm_weight=0.45,
            blend_threshold=0.6322, metrics{...}

data/artifacts/feature_lists/
├── top50_features_lgbm_train_feature_cv5.json    ← 50 feature adı (kaynak)
├── feature_importance_lgbm_top50_cv5.json        ← 50 feature gain/split değerleri
└── feature_importance_ensemble_top50_lstm.json   ← Ensemble efektif gain değerleri

src/training/
├── train_lgbm_top50_cv5.py                       ← LGBM eğitim scripti
├── train_lstm.py                                 ← LSTM OOF CV5 eğitim scripti
└── train_ensemble_top50_lstm.py                  ← Ensemble blending scripti

src/inference/
├── predictor.py                                  ← Model yükleme + inference engine
└── risk_scorer.py                                ← proba → risk_band + karar

api/
├── main.py                                       ← FastAPI app
├── schemas.py                                    ← Pydantic v2 request/response
├── validators.py                                 ← Field + cross-field kuralları
└── routers/ predict.py, explain.py, features.py

reports/
├── metrics_lgbm_top50_cv5_20260421.md
├── lstm_oof_cv5_report_20260422.md
├── ensemble_top50_lstm_report_20260422.md
└── ensemble_model_technical_summary.md           ← bu dosya
```

---

## 10. Teknik Notlar

### OpenMP çakışması (LightGBM + PyTorch)
LightGBM ve PyTorch aynı süreçte yüklendiğinde OpenMP runtime çakışması oluşur (exit code 139). Çözüm:
- `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` + `OMP_NUM_THREADS=1` ortam değişkenleri
- Her iki kütüphane **aynı anda** modül seviyesinde import edilmeli (önce torch, sonra lgbm)
- `torch.set_num_threads(1)` / `set_num_interop_threads(1)` try/except ile korunmalı

### LGBM NaN yönetimi
LGBM, NaN değerleri natively işler. Encoding sonrası `.fillna(0.0)` uygulamak AUC'u 0.878→0.819'a düşürür. Inference katmanında `fillna` **kullanılmaz**.

### OOF pozisyonel hizalama
LGBM train skorları ve LSTM OOF skorları aynı `train_test_split(test_size=0.2, random_state=42)` ile elde edilir. `customer_id` merge gerekmez; her iki array da 492.017 satır ve aynı sırada.

### Ensemble'da LGBM skoru yoksa
Sadece LSTM skoru (previous_applications verisi varsa) → fallback LGBM-only değil, LSTM tahmini doğrudan kullanılamaz (ağırlık optimizasyonu ikisi birlikte yapıldı). API bu durumda LGBM-only döner.

---

## 11. Yeniden Eğitim Talimatı

```bash
# 1. LGBM Top-50 CV5
python run.py --mode preprocess
python src/training/train_lgbm_top50_cv5.py

# 2. LSTM OOF CV5
python run.py --mode train-lstm

# 3. Ensemble (LGBM pkl + LSTM OOF npz yüklü olmalı)
python src/training/train_ensemble_top50_lstm.py

# 4. API sunucusu
python run.py --mode serve
```
