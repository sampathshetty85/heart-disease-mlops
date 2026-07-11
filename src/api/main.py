"""
main.py -- FastAPI application for Heart Disease Prediction.

Startup: loads pipeline.joblib once via load_artifacts() — zero per-request I/O.
Endpoints:
  GET  /health   -- liveness check (used by K8s probes in Phase 7)
  POST /predict  -- accepts PatientFeatures JSON, returns PredictionResponse
  GET  /metrics  -- Prometheus metrics (auto-exposed by instrumentator)

Note: feature_columns.json is generated at docker build time by preprocess.py.
      It must exist before the API starts. In Docker, this is guaranteed by the
      RUN preprocess.py build step. Locally, run preprocess.py first.
"""

import os
import sys
import json
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api.schemas import PatientFeatures, PredictionResponse  # noqa: E402
from models.predict import load_artifacts, predict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_model_name = "unknown"

PREDICTION_COUNTER = Counter(
    "heart_disease_predictions_total",
    "Total predictions by outcome label",
    ["label"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_name
    pipeline, _ = load_artifacts()
    _model_name = type(pipeline.named_steps["classifier"]).__name__
    logger.info(json.dumps({
        "event": "startup",
        "model": _model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }))
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts presence/absence of heart disease from patient features.",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    return {"status": "ok", "model": _model_name}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(patient: PatientFeatures):
    t0 = time.perf_counter()
    result = predict(patient.model_dump())
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    logger.info(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": "/predict",
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "label": result["label"],
        "latency_ms": latency_ms,
    }))

    PREDICTION_COUNTER.labels(label=result["label"]).inc()
    return result
