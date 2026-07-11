"""
test_api.py -- unit tests for FastAPI /health and /predict endpoints.

All tests are skipped until Phase 6 builds src/api/main.py.
Activated in Phase 6 by removing the skip markers.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

API_READY = True  # FastAPI app built in Phase 6


@pytest.mark.skipif(not API_READY, reason="FastAPI app built in Phase 6")
def test_health_endpoint():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200


@pytest.mark.skipif(not API_READY, reason="FastAPI app built in Phase 6")
def test_predict_endpoint_valid():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    payload = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


@pytest.mark.skipif(not API_READY, reason="FastAPI app built in Phase 6")
def test_predict_response_keys():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    payload = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }
    data = client.post("/predict", json=payload).json()
    assert "prediction" in data
    assert "confidence" in data


@pytest.mark.skipif(not API_READY, reason="FastAPI app built in Phase 6")
def test_predict_prediction_range():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    payload = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }
    data = client.post("/predict", json=payload).json()
    assert data["prediction"] in (0, 1)


@pytest.mark.skipif(not API_READY, reason="FastAPI app built in Phase 6")
def test_predict_confidence_range():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    payload = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }
    data = client.post("/predict", json=payload).json()
    assert 0.0 <= data["confidence"] <= 1.0


@pytest.mark.skipif(not API_READY, reason="FastAPI app built in Phase 6")
def test_predict_invalid_input():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    response = client.post("/predict", json={"age": 63})
    assert response.status_code == 422
