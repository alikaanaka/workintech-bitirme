# LSTM OOF CV5 Eğitim Raporu

**Tarih:** 20260422  
**Strateji:** 5-fold stratified OOF cross-validation  
**Mimari:** HybridLSTMClassifier (BiLSTM + ScaledDotAttention + LayerNorm)  
**Device:** cpu  

## Hiperparametreler

| Parametre | Değer |
|---|---|
| hidden_size | 128 |
| num_layers | 2 (bidirectional) |
| dropout | 0.3 |
| optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| loss | FocalLoss (gamma=2.0, pos_weight=neg/pos) |
| max_epochs | 50 |
| early_stopping_patience | 6 |
| gradient_clip_max_norm | 1.0 |
| SEQUENCE_MAX_LEN | 15 |
| batch_size | 512 |
| n_folds | 5 |

## Fold Bazlı Sonuçlar

| Fold | OOF AUC | Best Val AUC | Best Epoch |
|---:|---:|---:|---:|
| 1 | 0.873985 | 0.873985 | 50 |
| 2 | 0.875096 | 0.875096 | 49 |
| 3 | 0.874677 | 0.874677 | 50 |
| 4 | 0.872280 | 0.872280 | 48 |
| 5 | 0.871249 | 0.871249 | 49 |
| **Mean** | **0.873457** | — | — |
| **Std** | **0.001464** | — | — |

## OOF Genel Metrikler (tüm 5 fold birleşik)

| Metrik | Değer |
|---|---:|
| OOF AUC (overall) | 0.873347 |
| OOF F1-optimal threshold | 0.6120 |
| OOF F1 @ optimal threshold | 0.531027 |

## Final Model — Holdout Metrikleri

Final model tüm train seti üzerinde eğitildi (erken durdurma için %10 iç val ayrıldı).  
**Best epoch:** 50 | **Best val AUC (iç):** 0.892441  

| Metrik | Değer |
|---|---:|
| Holdout AUC | 0.891374 |
| Holdout Gini | 0.782747 |
| Holdout KS | 0.648066 |
| Holdout PR-AUC | 0.587587 |
| Holdout F1 @ 0.6086 | 0.548078 |
| Holdout Brier Score | 0.112743 |

**Confusion Matrix (holdout):** `[[108464, 4611], [4441, 5489]]`

## Kaydedilen Dosyalar

- Model: `/Users/alikaanaka/Downloads/workintech-bitirme/models_saved/lstm_oof_cv5_20260422.pt`
- OOF predictions: `/Users/alikaanaka/Downloads/workintech-bitirme/models_saved/lstm_oof_predictions_20260422.npz`

## Teknik Notlar

- OOF tahminleri her fold'da o fold'un kendi encoder/scaler/categorical_maps'i ile üretildi.
- Final model, tüm train_full verisi üzerinde bağımsız olarak eğitildi.
- OOF tahminleri stacking meta-model için kullanıma hazır.
- Geçmişsiz müşteri: mask tamamen sıfır → _ScaledDotAttention uniform attention → static feature tabanlı tahmin.