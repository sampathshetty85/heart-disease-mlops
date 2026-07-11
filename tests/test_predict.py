"""
test_predict.py -- unit tests for src/models/predict.py (Gap 36).

Tests the inference module directly without a running server,
so these run in CI before Phase 6 FastAPI app is built.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

MODELS_READY = all(
    os.path.exists(os.path.join(MODELS_DIR, f))
    for f in ["model.joblib", "pipeline.joblib"]
)

SAMPLE_INPUT = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
}


@pytest.mark.skipif(not MODELS_READY, reason="Run src/models/train.py first")
def test_predict_returns_dict():
    from models.predict import predict
    result = predict(SAMPLE_INPUT)
    assert isinstance(result, dict)


@pytest.mark.skipif(not MODELS_READY, reason="Run src/models/train.py first")
def test_predict_keys_present():
    from models.predict import predict
    result = predict(SAMPLE_INPUT)
    assert "prediction" in result
    assert "confidence" in result
    assert "label" in result


@pytest.mark.skipif(not MODELS_READY, reason="Run src/models/train.py first")
def test_predict_prediction_range():
    from models.predict import predict
    result = predict(SAMPLE_INPUT)
    assert result["prediction"] in (0, 1)


@pytest.mark.skipif(not MODELS_READY, reason="Run src/models/train.py first")
def test_predict_confidence_range():
    from models.predict import predict
    result = predict(SAMPLE_INPUT)
    assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.skipif(not MODELS_READY, reason="Run src/models/train.py first")
def test_predict_determinism():
    from models.predict import predict
    r1 = predict(SAMPLE_INPUT)
    r2 = predict(SAMPLE_INPUT)
    assert r1["prediction"] == r2["prediction"]
    assert r1["confidence"] == r2["confidence"]
