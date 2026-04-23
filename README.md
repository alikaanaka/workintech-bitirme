# Home Credit Risk System - Phase 1

Bu fazın amacı, model eğitiminden önce güvenilir bir preprocess ve feature dataset altyapısı kurmaktır.

## Veri Kaynakları

- `data/raw/train_merged.csv`
- `data/raw/previous_application.csv`

Repository uyumluluğu için geçici fallback dosya adları da desteklenir:
- `data/train_feature.csv`
- `data/prev_app_customer_level.csv`

## Preprocess Akışı

1. Veri dosyaları yüklenir ve path doğrulanır.
2. Minimum schema kontrolleri uygulanır (`SK_ID_CURR`, `TARGET`, vb.).
3. Temizleme adımları çalışır:
   - `DAYS_EMPLOYED == 365243` -> `NaN`
   - `DAYS_*` pozitif değerler -> `NaN`
   - `AMT_INCOME_TOTAL` winsorize (99.9p)
   - kategorik `XNA` / `XAP` -> `NaN`
   - `SELLERPLACE_AREA == -1` -> `NaN`
4. Ana tablo feature'ları üretilir (oranlar, EXT_SOURCE özetleri, document toplamı vb.).
5. `previous_application` müşteri bazında aggregate edilir.
6. Ana tablo ile aggregate tablo `SK_ID_CURR` üzerinden left join edilir.
7. `NO_PREV_APP_FLAG` üretilir ve çıktı kaydedilir.

## Çalıştırma

```bash
python run.py --mode preprocess
```

Bu komut preprocess pipeline'ı uçtan uca çalıştırır ve:
- `data/processed/final_dataset.parquet` dosyasını üretir
- satır/sütun özetini loglar

## Not

- Bu fazda model eğitimi, API, UI, SHAP ve sequence/LSTM çıktıları yoktur.
- Eğer gerçek `previous_application.csv` yerine müşteri seviyesinde özet bir dosya kullanılırsa,
  bazı aggregation feature'ları kısıtlı bilgiyle üretilir.

## Phase 2 - LightGBM Baseline

LightGBM baseline egitimi `final_dataset.parquet` uzerinden calisir.

### Egitim Komutu

```bash
python run.py --mode train-lgbm
```

Istege bagli rapor kontrolu:

```bash
python run.py --mode evaluate
```

### Uretilen Artefact'lar

- `models_saved/lgbm_model.pkl`: egitilmis LightGBM + encoder paketi
- `reports/metrics_comparison.md`: validation ve holdout metrik ozeti
- `data/artifacts/feature_lists/top50_features.json`: gain importance bazli ilk 50 feature

### top50_features.json Ne Icin Kullanilir?

- Model davranisini hizli analiz etmek
- UI/API katmanlari icin oncelikli feature listesi saglamak
- Sonraki fazlarda SHAP ve feature governance adimlarina temel olmak

### Model Sonuc Kayitlari

- Ozet metrik raporu: `reports/metrics_comparison.md`
- Kalici sonuc kaydi: `reports/model_results.md`

## Phase 3 - LSTM Branch + Ensemble

Phase 3 ile previous_application tablosundan sequence veri uretilir ve static + sequence hibrit LSTM modeli egitilir.

### Inference Modlari (Kritik Kural)

- Dataset-backed inference:
  - previous_application erisimi vardir
  - sequence uretilir
  - LightGBM + LSTM skor ensemble kullanilir
- Manual form inference:
  - sequence veri yoktur
  - LSTM skoru uretilmez
  - sadece LightGBM skoru kullanilir

### Egitim Komutu

```bash
python run.py --mode train-lstm --threshold 0.30
```

### Uretilen Artefact'lar

- `models_saved/lstm_model.pt`
- `models_saved/model_metadata.json`
