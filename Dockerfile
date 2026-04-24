FROM python:3.11-slim

WORKDIR /app

# LightGBM için sistem bağımlılığı
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU-only — CUDA versiyonu ~3GB, CPU versiyonu ~200MB
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Kalan bağımlılıklar
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Uygulama kodu ve modeller
COPY api/     ./api/
COPY src/     ./src/
COPY config/  ./config/
COPY models_saved/ ./models_saved/

# LightGBM + PyTorch OpenMP çakışma çözümü
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV OMP_NUM_THREADS=1

# Cloud Run PORT değişkenini kullanır (varsayılan 8080)
ENV PORT=8080
EXPOSE 8080

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
