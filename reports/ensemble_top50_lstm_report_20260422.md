# Ensemble Raporu — LightGBM Top-50 CV5 + LSTM OOF CV5

**Tarih:** 20260422  
**Yöntem:** Score blending (ağırlıklı ortalama)  
**LGBM modeli:** `lgbm_top50_cv5_threshold030_20260421.pkl`  
**LSTM modeli:** `lstm_oof_cv5_20260422.pt`  
**LGBM ağırlığı (α):** `0.55` | **LSTM ağırlığı (1-α):** `0.45`  
**Ensemble formülü:** `0.55 × lgbm_score + 0.45 × lstm_score`  
**Threshold:** `0.6322` (OOF üzerinde F1 optimize edildi)  

## OOF Ağırlık Seçimi (train_full — leak-free)

| Metrik | Değer |
|---|---:|
| OOF Blend F1 @ 0.6322 | 0.651586 |
| LGBM weight grid searched | 0.00 → 1.00 (step=0.05) |

## Holdout Karşılaştırması

| Model | AUC | Gini | KS | PR-AUC | F1 | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| LGBM Top-50 standalone | 0.878313 | 0.756627 | 0.621884 | 0.433122 | 0.471461 | 0.6322 |
| LSTM OOF CV5 standalone | 0.891374 | 0.782747 | 0.648066 | 0.587587 | 0.546264 | 0.6322 |
| **Ensemble (α=0.55)** | **0.911442** | **0.822885** | **0.688893** | **0.609981** | **0.570417** | **0.6322** |

## Holdout Confusion Matrix

**Ensemble:** `[[108469, 4606], [4130, 5800]]`  
**LGBM standalone:** `[[104460, 8615], [4210, 5720]]`  
**LSTM standalone:** `[[109619, 3456], [4900, 5030]]`  

## Kaydedilen Dosyalar

`/Users/alikaanaka/Downloads/workintech-bitirme/models_saved/ensemble_top50_lstm_20260422.pkl`