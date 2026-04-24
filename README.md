# Home Credit Risk — Kredi Temerrüt Risk Tahmin Sistemi

Akbank temalı, uçtan uca kredi başvurusu risk değerlendirme sistemi. LightGBM + LSTM hibrit ensemble modeliyle gerçek zamanlı temerrüt riski tahmini yapar.

**Canlı Demo:** https://home-credit-493207.web.app  
**API:** https://home-credit-api-950128241973.europe-west1.run.app  
**API Docs:** https://home-credit-api-950128241973.europe-west1.run.app/docs

---

## İçindekiler

- [Proje Özeti](#proje-özeti)
- [Model Performansı](#model-performansı)
- [Sistem Mimarisi](#sistem-mimarisi)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Eğitim Pipeline](#eğitim-pipeline)
- [API Kullanımı](#api-kullanımı)
- [Frontend](#frontend)
- [Deploy](#deploy)

---

## Proje Özeti

Home Credit Group veri seti üzerinde geliştirilen bu sistem, kredi başvurularının temerrüt riskini üç katmanlı bir yaklaşımla değerlendirir:

1. **LightGBM** — Top-50 özellik üzerinde gradient boosting, hızlı ve yorumlanabilir
2. **LSTM** — Önceki başvuru geçmişinden sequence öğrenimi (HybridLSTMClassifier)
3. **Ensemble** — LGBM (%55) + LSTM (%45) ağırlıklı birleşim

Sistem iki inference modunu destekler:
- **Manuel form**: Kullanıcı verileri girer → yalnızca LGBM çalışır
- **Sequence modu**: Önceki başvurular eklenir → LGBM + LSTM + Ensemble çalışır

---

## Model Performansı

| Model | Holdout AUC | Karar Eşiği |
|-------|-------------|-------------|
| LightGBM (Top-50, CV5) | **0.8783** | 0.30 |
| LSTM (HybridLSTM, CV5 OOF) | **0.8914** | 0.6086 |
| Ensemble (0.55×LGBM + 0.45×LSTM) | **0.9114** | 0.6322 |

---

## Sistem Mimarisi

```
┌─────────────────────┐     HTTPS      ┌──────────────────────────────┐
│   Firebase Hosting  │ ◄────────────► │        Kullanıcı Tarayıcı    │
│  (HTML/CSS/JS)      │                │  akbank-homepage.html        │
│  home-credit-       │                │  credit_risk_dashboard_      │
│  493207.web.app     │                │  akbank.html                 │
└─────────────────────┘                │  homecredit_project_         │
                                       │  details.html                │
                                       └──────────────┬───────────────┘
                                                      │ POST /api/predict
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │   Google Cloud Run           │
                                       │   FastAPI + Uvicorn          │
                                       │   europe-west1               │
                                       │                              │
                                       │  ┌─────────┐ ┌──────────┐  │
                                       │  │ LightGBM│ │   LSTM   │  │
                                       │  │  Top-50 │ │ Hybrid   │  │
                                       │  └────┬────┘ └────┬─────┘  │
                                       │       └─────┬──────┘        │
                                       │        ┌────▼─────┐         │
                                       │        │ Ensemble │         │
                                       │        └──────────┘         │
                                       └──────────────────────────────┘
```

---

## Proje Yapısı

```
workintech-bitirme/
├── api/                        # FastAPI uygulaması
│   ├── main.py                 # CORS, middleware, router kayıtları
│   ├── config.py               # model_config.yaml okuyucu
│   ├── schemas.py              # Pydantic request/response modelleri
│   ├── validators.py           # Alan ve çapraz doğrulamalar
│   └── routers/
│       ├── predict.py          # POST /api/predict
│       ├── explain.py          # POST /api/explain
│       └── features.py         # GET /api/features
│
├── src/
│   ├── data/                   # Veri yükleme ve temizleme
│   ├── features/               # Feature mühendisliği pipeline
│   ├── models/
│   │   └── lstm_model.py       # HybridLSTMClassifier
│   ├── training/               # Eğitim scriptleri
│   │   ├── train_lgbm_top50_cv5.py
│   │   └── train_lstm.py
│   ├── inference/
│   │   ├── predictor.py        # ModelPredictor singleton
│   │   └── risk_scorer.py      # Risk bandı hesaplama
│   └── preprocessing/
│       └── encoder.py          # FoldEncoder
│
├── config/
│   └── model_config.yaml       # Eşikler, ağırlıklar, AUC değerleri
│
├── models_saved/               # Eğitilmiş model artefaktları
│   ├── lgbm_top50_cv5_threshold030_20260421.pkl
│   ├── lstm_oof_cv5_20260422.pt
│   ├── ensemble_top50_lstm_20260422.pkl
│   └── model_metadata.json
│
├── frontend/
│   ├── akbank-homepage.html          # Akbank ana sayfa (slider, hızlı işlemler)
│   ├── credit_risk_dashboard_akbank.html  # Risk değerlendirme dashboard
│   └── homecredit_project_details.html   # Proje teknik detayları
│
├── tests/                      # Pytest test suite
├── Dockerfile                  # Cloud Run image tanımı
├── requirements-prod.txt       # Production bağımlılıkları
├── requirements.txt            # Geliştirme bağımlılıkları
└── run.py                      # CLI eğitim komutu
```

---

## Kurulum

### Gereksinimler

- Python 3.11+
- pip

### Yerel Geliştirme

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# API'yi başlat
uvicorn api.main:app --reload --port 8000
```

API `http://localhost:8000` adresinde çalışır. Swagger UI: `http://localhost:8000/docs`

### Frontend

Frontend saf HTML/CSS/JS dosyalarıdır, hiçbir build adımı gerekmez.

```bash
# Herhangi bir HTTP sunucusuyla servis edilebilir
cd frontend
python -m http.server 3000
```

---

## Eğitim Pipeline

```bash
# 1. Preprocess
python run.py --mode preprocess

# 2. LightGBM eğitimi (Top-50, CV5)
python run.py --mode train-lgbm

# 3. LSTM eğitimi
python run.py --mode train-lstm --threshold 0.30

# 4. Değerlendirme
python run.py --mode evaluate
```

### Üretilen Artefaktlar

| Dosya | Açıklama |
|-------|----------|
| `models_saved/lgbm_top50_cv5_threshold030_20260421.pkl` | LightGBM + encoder |
| `models_saved/lstm_oof_cv5_20260422.pt` | LSTM model ağırlıkları |
| `models_saved/ensemble_top50_lstm_20260422.pkl` | Ensemble metadata |
| `data/artifacts/feature_lists/top50_features_lgbm_train_feature_cv5.json` | Top-50 feature listesi |

---

## API Kullanımı

### Sağlık Kontrolü

```bash
GET /api/health
```

```json
{
  "status": "ok",
  "models_loaded": ["lgbm", "lstm", "ensemble"],
  "auc_holdout": { "lgbm": 0.8783, "lstm": 0.8914, "ensemble": 0.9114 }
}
```

### Risk Tahmini

```bash
POST /api/predict
Content-Type: application/json
```

```json
{
  "EXT_SOURCE_1": 0.6,
  "EXT_SOURCE_2": 0.55,
  "EXT_SOURCE_3": 0.5,
  "AMT_CREDIT": 200000,
  "AMT_ANNUITY": 10000,
  "AMT_INCOME_TOTAL": 90000,
  "DAYS_BIRTH": -12000,
  "DAYS_EMPLOYED": -2000,
  "previous_applications": [
    {
      "AMT_CREDIT": 150000,
      "AMT_ANNUITY": 8000,
      "NAME_CONTRACT_TYPE": "Cash loans",
      "NAME_CONTRACT_STATUS": "Approved",
      "DAYS_DECISION": -730
    }
  ]
}
```

**Yanıt:**

```json
{
  "proba_lgbm": 0.031,
  "proba_lstm": 0.008,
  "proba_ensemble": 0.015,
  "risk_score_pct": 3,
  "risk_band": "Low",
  "decision": "ONAYLA",
  "threshold_used": 0.6322,
  "available_models": ["lgbm", "lstm", "ensemble"]
}
```

> `previous_applications` gönderilmezse yalnızca LGBM çalışır. Gönderilirse LGBM + LSTM + Ensemble devreye girer.

### Diğer Endpointler

| Endpoint | Metot | Açıklama |
|----------|-------|----------|
| `/api/predict` | POST | Risk tahmini |
| `/api/explain` | POST | Feature katkı açıklaması |
| `/api/features` | GET | Top-50 feature listesi ve önem skorları |
| `/api/health` | GET | Model yükleme durumu ve AUC değerleri |

---

## Frontend

### Sayfalar

| Sayfa | URL | Açıklama |
|-------|-----|----------|
| Ana Sayfa | `/akbank-homepage.html` | Akbank temalı giriş sayfası |
| Risk Dashboard | `/credit_risk_dashboard_akbank.html` | 50 alan, 3 demo profil, gauge |
| Proje Detayları | `/homecredit_project_details.html` | Teknik mimari ve model açıklamaları |

### Dashboard Özellikleri

- **50 feature** accordion gruplarında (EXT Source, Kredi, Günler, Büro, Taksit, KK, Demografik)
- **Otomatik türetilen alanlar**: EXT_SOURCE_MEAN, CREDIT_TERM, ANNUITY_INCOME_RATIO, CREDIT_INCOME_RATIO, DAYS_EMPLOYED_PERCENT, b_total_loan_count
- **3 demo profil**: Az Risk (~skor 2), Orta Risk (~skor 43), Yüksek Risk (~skor 80)
- **LSTM modu**: Önceki başvurular toggle'ı açıldığında otomatik demo veri dolar, ensemble aktif olur
- **Sonuç paneli**: SVG gauge, model olasılık çubukları, karar kutusu

---

## Deploy

### Firebase Hosting (Frontend)

```bash
# Firebase CLI kurulumu
npm install -g firebase-tools
firebase login

# Deploy
firebase deploy --only hosting
```

### Google Cloud Run (API)

```bash
# Docker image build (Google Cloud Build)
gcloud builds submit --tag gcr.io/home-credit-493207/home-credit-api --timeout=20m

# Cloud Run deploy
gcloud run deploy home-credit-api \
  --image gcr.io/home-credit-493207/home-credit-api \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 60 \
  --max-instances 3
```

### Ortam Değişkenleri (Dockerfile)

| Değişken | Değer | Açıklama |
|----------|-------|----------|
| `PORT` | `8080` | Cloud Run port |
| `KMP_DUPLICATE_LIB_OK` | `TRUE` | LightGBM + PyTorch OpenMP çakışma çözümü |
| `OMP_NUM_THREADS` | `1` | OpenMP thread sayısı |

---

## Teknoloji Yığını

| Katman | Teknoloji |
|--------|-----------|
| ML | LightGBM 4.6, PyTorch 2.11 (CPU), scikit-learn 1.8 |
| API | FastAPI 0.136, Uvicorn, Pydantic v2, slowapi |
| Veri | pandas 3.0, numpy 2.4, pyarrow 24 |
| Frontend | Vanilla HTML/CSS/JS (dependency yok) |
| Hosting | Firebase Hosting (ücretsiz tier) |
| API Hosting | Google Cloud Run (europe-west1) |
| CI/CD | Google Cloud Build |
