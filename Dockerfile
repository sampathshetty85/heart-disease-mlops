FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer — only rebuilds when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and static assets
COPY src/        ./src/
COPY notebooks/  ./notebooks/
COPY output/     ./output/
COPY screenshots/ ./screenshots/

# Create data and model directories (populated by RUN steps below)
RUN mkdir -p data/raw data/processed models

# ---------------------------------------------------------------------------
# Build-time pipeline — runs ONCE per docker build
# Fetches live UCI data, trains fresh model, bakes artifacts into image layer
# ---------------------------------------------------------------------------
RUN python src/data/download.py
RUN python src/data/preprocess.py
RUN python src/models/train.py
RUN python src/models/package.py

# ---------------------------------------------------------------------------
# Runtime — executes on every docker run / container restart
# Model is already trained — startup is instant, no retraining
# ---------------------------------------------------------------------------
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
