# Oturum Notu — LSTM F1 İyileştirme & Ensemble Tartışması

**Tarih:** 2026-04-21  
**Konu:** LSTM modelinin F1 düşüşünün analizi, mimarinin yeniden tasarımı, eğitim ve stacking planı

---

## 1. Başlangıç Durumu

`reports/metrics_comparison.md` dosyasındaki Phase 3 sonuçları:

| Model | AUC | F1 | Threshold |
|---|---|---|---|
| LightGBM (Phase 2) | 0.893654 | 0.281332 | 0.22 |
| LSTM v1 | ~0.531 | 0.149403 | 0.30 |
| Ensemble v1 | ~0.934 | 0.149403 | 0.30 |

LSTM confusion matrix: TP=9930, FP=113069, TN=6, FN=0 — model neredeyse herkesi pozitif etiketliyordu.

---

## 2. Tespit Edilen Kök Nedenler

1. **Sequence featureları normalize edilmemişti** — AMT_CREDIT (milyonlar), DAYS_DECISION (-2000) gibi wildly farklı ölçekler LSTM'i körleştiriyordu. LSTM AUC ~0.50 (rastgele).
2. **Static featurelar normalize edilmemişti** — Linear projection katmanları için kritik.
3. **Mean pooling** — Tüm timestep çıktılarının ortalaması önemli olayların sinyal ağırlığını azaltıyordu.
4. **Threshold sabit 0.30** — F1 için optimize edilmemişti.
5. **Ensemble ağırlığı AUC'a göre seçiliyordu** — LSTM rastgele skor üretirken %95 LSTM / %5 LGBM seçildi, F1 çöktü.

---

## 3. Yapılan İyileştirmeler

### `src/models/lstm_model.py`
- `SEQUENCE_MAX_LEN` 10 → 15
- Mean pooling yerine **`_ScaledDotAttention`** (learnable query, masked softmax)
- Sequence projection branch'e **`nn.LayerNorm`** eklendi
- Static projection branch'e **`nn.LayerNorm`** eklendi

### `src/training/train_lstm.py`
- **Sequence StandardScaler**: Sadece padding olmayan pozisyonlara fit, padding sonrası sıfırlama
- **Static StandardScaler**: FoldEncoder çıktısına uygulandı
- **`_FocalLoss(gamma=2.0)`**: BCEWithLogitsLoss yerine; zor örneklere odaklanma
- **Gradient clipping** `max_norm=1.0`
- **`_find_f1_optimal_threshold()`**: Precision-recall curve üzerinden F1 maximize eden threshold
- **`_select_lgbm_weight_by_f1()`**: AUC yerine F1 üzerinde grid search ile blend ağırlığı
- Model checkpoint'e `static_scaler` ve `seq_scaler` eklendi

---

## 4. Eğitim Sonuçları

**Hiperparametreler:**
- hidden_size: 128, num_layers: 2 (bidirectional), dropout: 0.3
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Loss: FocalLoss (gamma=2.0, pos_weight=neg/pos ratio)
- max_epochs: 50, early_stopping_patience: 6
- batch_size: 512, device: cpu
- Best epoch: 49, Best validation AUC: 0.8669

**AUC eğitim seyri (seçilmiş epoch'lar):**

| Epoch | Val AUC |
|---|---|
| 1 | 0.769 |
| 5 | 0.780 |
| 10 | 0.791 |
| 20 | 0.826 |
| 30 | 0.850 |
| 40 | 0.860 |
| 49 | 0.867 |

**F1-optimal threshold:** 0.6020  
**Seçilen LightGBM weight:** 0.35 (LSTM: 0.65)

**Nihai metrikler (holdout):**

| Model | AUC | F1 | Threshold |
|---|---|---|---|
| LSTM v2 | 0.869516 | 0.525940 | 0.6020 |
| Ensemble v2 | 0.895893 | 0.579880 | 0.6020 |

**LSTM F1: 0.149 → 0.526 (+252%) | Ensemble F1: 0.149 → 0.580 (+289%)**

Detaylı rapor: `reports/lstm_f1_improvement_report.md`

---

## 5. Top 50 Feature (Ensemble LightGBM Bileşeni)

`data/artifacts/feature_lists/top50_features_ensemble_lgbm_lstm.json` dosyasına kaydedildi.

**Öne çıkan gruplar:**

| Grup | Örnekler | Anlam |
|---|---|---|
| Dış kredi skoru | EXT_SOURCE_MEAN/1/2/3, PROD, STD | 3 farklı kredi bürosundan dış skorlar |
| Türetilmiş oranlar | CREDIT_TERM, ANNUITY_INCOME_RATIO, DAYS_EMPLOYED_PERCENT | Borç yükü ve istikrar oranları |
| Gün featureları | DAYS_EMPLOYED, DAYS_BIRTH, DAYS_LAST_PHONE_CHANGE | Zaman bazlı istikrar sinyalleri |
| `b_` (bureau) | b_avg_utilization, b_total_current_debt | Geçmiş kredi kullanım davranışı |
| `int_` (taksit) | int_max_ins_days_late_ever, int_avg_payment_performance | Taksit ödeme geçmişi |
| `cc_` (kredi kartı) | cc_total_avg_utilization_ratio, cc_avg_repayment_performance | Kart kullanım davranışı |

---

## 6. Ensemble Mimarisi (Mevcut)

Şu an uygulanan yöntem **score blending** (skor karıştırma):

```
LightGBM  →  P(default) skoru ─┐
                                 ├→  0.35 × lgbm + 0.65 × lstm  →  final skor
LSTM      →  P(default) skoru ─┘
```

İki model birbirinden bağımsız eğitildi. LSTM çıktıları LightGBM'e feature olarak **girmedi**.

---

## 7. Planlanan: Stacking Modeli

Hedef mimari:

```
Static features   →  LightGBM (OOF)  →  skor_lgbm ─┐
                                                       ├→  Meta-model
Sequence features →  LSTM (OOF)      →  skor_lstm ─┘
```

**Teknik not:** Meta-model eğitiminde her iki modelin skorları **out-of-fold (OOF)** tahminlerden gelmeli — aksi hâlde model kendi eğitim verisini ezberlemiş olur ve meta-model leak'li veriden öğrenir.

Şu an LightGBM'in CV5 modeli mevcut (OOF üretilebilir), LSTM tek fold. Başlamadan önce LSTM için de OOF stratejisi planlanmalı.

**Durum:** Henüz uygulanmadı — onay bekleniyor.

---

## 8. Geçmişsiz Müşteri Fallback Tartışması

LSTM sequence-tabanlı bir model olduğundan `previous_application` kaydı olmayan müşteriler için fallback gerekiyor.

**Mevcut durum:** Geçmişsiz müşteride `sequences` ve `mask` tamamen sıfır. `_ScaledDotAttention` bunu yakalıyor (uniform attention → sıfır vektör), model sadece static featurelarla tahmin üretiyor. Çıktı 0.5 değil, static bilgiye göre gerçek tahmin.

**Tartışılan seçenekler:**

| Seçenek | Davranış | Not |
|---|---|---|
| 0.5 fallback | LSTM skoru sabit 0.5, ensemble devam eder | Tüm geçmişsiz müşterilere sabit offset ekler |
| Sadece LightGBM | LSTM devre dışı, sadece LGBM skoru | Manual-form moduna eşdeğer |
| Mevcut durum | Static featurelarla LSTM yine de tahmin üretir | Sequence katkısı sıfır |

**Durum:** Karar bekleniyor.
