# LSTM F1 İyileştirme Raporu

## Sorun Analizi (Önceki Durum)
- LSTM Holdout AUC ~0.50 (rastgele tahmin seviyesi)
- Ensemble F1 @ threshold 0.30: 0.149 (LightGBM standalone F1: 0.281)
- Confusion matrix: TP=9930, FP=113069, TN=6, FN=0 — model neredeyse herkesi pozitif etiketliyordu

## Tespit Edilen Kök Nedenler
1. **Sequence featureları normalize edilmemişti**: AMT_CREDIT (milyonlar), DAYS_DECISION (-2000) gibi
   çok farklı ölçeklerdeki ham değerler LSTM'in öğrenmesini engelliyordu.
2. **Static featurelar da ölçeklenmemişti**: Linear projection katmanları için kritik.
3. **Mean pooling**: Tüm timestep çıktılarının ortalaması önemli application olaylarının
   sinyal ağırlığını azaltıyordu.
4. **Threshold 0.30 sabit kullanıldı**: F1 için optimal değildi.
5. **Ensemble ağırlığı AUC'a göre seçildi**: LSTM rastgele skorlar üretirken
   %95 LSTM ağırlığı (%5 LightGBM) seçildi — F1 çöktü.

## Yapılan İyileştirmeler

### 1. Sequence Feature Standardizasyonu
- Sadece gerçek (padding olmayan) pozisyonlar üzerinde `StandardScaler` fit edildi.
- AMT_CREDIT, AMT_APPLICATION, AMT_ANNUITY, CNT_PAYMENT, DAYS_DECISION
  sıfır ortalama, birim varyans ölçeğine getirildi.
- Padding pozisyonları ölçekleme sonrasında tekrar sıfırlandı.

### 2. Static Feature Standardizasyonu
- `FoldEncoder` sonrası static featurelar `StandardScaler` ile normalize edildi.
- Nöral ağ lineer katmanları için kritik; LightGBM bu adıma ihtiyaç duymuyor.

### 3. Attention Pooling
- Mean pooling yerine learnable query ile scaled dot-product attention eklendi.
- `_ScaledDotAttention`: Modelin en bilgilendirici previous application olayına
  odaklanmasını sağlıyor.
- Padding maskesi `-inf` ile attention skorlarından dışlanıyor.

### 4. LayerNorm
- Sequence projection branch'e `nn.LayerNorm` eklendi.
- Static projection branch'e `nn.LayerNorm` eklendi.
- Gradient akışını ve eğitim stabilitesini iyileştiriyor.

### 5. Focal Loss (gamma=2.0)
- `BCEWithLogitsLoss` yerine Focal Loss kullanıldı.
- Kolay örneklere (çoğunluk sınıfı) verilen ağırlık azaltılarak
  zor örneklere (azınlık — default) odaklanma sağlandı.
- Sınıf dengesizliği için pos_weight korundu.

### 6. Gradient Clipping (max_norm=1.0)
- `nn.utils.clip_grad_norm_` ile gradient patlaması önlendi.
- Ölçeklenmemiş girdi nedeniyle oluşan gradient instabilitesi giderildi.

### 7. F1-Optimal Threshold Arama
- Validation seti üzerinde precision-recall curve'den F1 maksimize eden
  threshold arandı.
- Sabit 0.30 yerine veri odaklı threshold kullanıldı.

### 8. F1-Tabanlı Ensemble Ağırlık Seçimi
- `select_lgbm_weight_by_auc` yerine `_select_lgbm_weight_by_f1` ile
  F1 @ f1_threshold üzerinde grid search yapıldı.
- Artık ensemble ağırlığı F1 metriğini maksimize ediyor.

### 9. Sequence Uzunluğu Artırıldı
- `SEQUENCE_MAX_LEN` 10'dan 15'e çıkarıldı.
- Daha uzun uygulama geçmişi sinyali yakalanıyor.

## Sonuçlar

### LSTM (Standalone — Holdout)
- AUC: 0.869516
- Gini: 0.739032
- KS: 0.605282
- PR-AUC: 0.548681
- F1 @ threshold 0.6020: 0.525940
- Confusion Matrix: TP=5307, FP=4944, TN=108131, FN=4623

### Ensemble (LGBM + LSTM — Holdout)
- AUC: 0.895893
- Gini: 0.791785
- KS: 0.660694
- PR-AUC: 0.614308
- F1 @ threshold 0.6020: 0.579880
- Confusion Matrix: TP=6087, FP=4977, TN=108098, FN=3843
- LightGBM weight: 0.35 | LSTM weight: 0.65

## Karşılaştırma Özeti

| Model | AUC | F1 | Threshold |
|---|---|---|---|
| LightGBM (Phase 2) | 0.893654 | 0.281332 | 0.22 |
| LSTM v1 (önceki) | ~0.531 | 0.149403 | 0.30 |
| Ensemble v1 (önceki) | ~0.934 | 0.149403 | 0.30 |
| LSTM v2 (bu çalışma) | 0.869516 | 0.525940 | 0.6020 |
| Ensemble v2 (bu çalışma) | 0.895893 | 0.579880 | 0.6020 |

## Kullanılan Hiperparametreler

- hidden_size: 128
- num_layers: 2 (bidirectional)
- dropout: 0.3
- optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- loss: FocalLoss (gamma=2.0, pos_weight=neg/pos ratio)
- max_epochs: 50, early_stopping_patience: 6
- gradient_clip_max_norm: 1.0
- SEQUENCE_MAX_LEN: 15
- batch_size: 512
- Best epoch: 49
- Best validation AUC: 0.866900
- F1-optimal threshold: 0.6020
- Selected LightGBM weight: 0.35
- Device: cpu