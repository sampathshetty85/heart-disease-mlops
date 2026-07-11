"""
test_model.py -- unit tests for model artifacts and inference pipeline.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

MODELS_READY = all(
    os.path.exists(os.path.join(MODELS_DIR, f))
    for f in ["model.joblib", "pipeline.joblib"]
)
DATA_READY = all(
    os.path.exists(os.path.join(DATA_DIR, f))
    for f in ["X_test.csv", "y_test.csv"]
)

SAMPLE_INPUT = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
}


@pytest.mark.skipif(not MODELS_READY, reason="Run src/models/train.py first")
def test_pipeline_loads(pipeline):
    assert pipeline is not None


@pytest.mark.skipif(not MODELS_READY, reason="Run src/models/train.py first")
def test_model_loads(model):
    assert model is not None


@pytest.mark.skipif(not MODELS_READY, reason="Run src/models/train.py first")
def test_pipeline_has_steps(pipeline):
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


@pytest.mark.skipif(not MODELS_READY, reason="Run src/models/train.py first")
def test_prediction_output_type(pipeline, X_test):
    preds = pipeline.predict(X_test)
    assert all(isinstance(int(p), int) for p in preds)


@pytest.mark.skipif(not MODELS_READY, reason="Run src/models/train.py first")
def test_prediction_output_range(pipeline, X_test):
    preds = pipeline.predict(X_test)
    assert set(preds).issubset({0, 1})


@pytest.mark.skipif(not MODELS_READY, reason="Run src/models/train.py first")
def test_confidence_range(pipeline, X_test):
    probas = pipeline.predict_proba(X_test)
    assert probas.shape[1] == 2
    assert (probas >= 0).all() and (probas <= 1).all()


@pytest.mark.skipif(not (MODELS_READY and DATA_READY), reason="Run train.py and preprocess.py first")
def test_roc_auc_above_threshold(pipeline, X_test, y_test):
    from sklearn.metrics import roc_auc_score
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    assert auc > 0.5, f"ROC-AUC {auc:.4f} not above 0.5"


@pytest.mark.skipif(not MODELS_READY, reason="Run src/models/train.py first")
def test_predict_dict_keys():
    from models.predict import predict
    result = predict(SAMPLE_INPUT)
    assert "prediction" in result
    assert "confidence" in result
    assert "label" in result
