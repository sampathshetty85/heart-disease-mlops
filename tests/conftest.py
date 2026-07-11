"""
conftest.py -- session-scoped fixtures shared across all test modules.

Data-dependent fixtures skip with a clear message if processed files
don't exist (e.g. preprocess.py hasn't been run yet).
Model-dependent fixtures skip if models/ artifacts don't exist.
"""

import os
import pytest
import pandas as pd

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

DATA_READY = all(
    os.path.exists(os.path.join(DATA_DIR, f))
    for f in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "feature_columns.json"]
)

MODELS_READY = all(
    os.path.exists(os.path.join(MODELS_DIR, f))
    for f in ["model.joblib", "pipeline.joblib"]
)

RAW_READY = os.path.exists(os.path.join(REPO_ROOT, "data", "raw", "heart.csv"))


@pytest.fixture(scope="session")
def X_train():
    if not DATA_READY:
        pytest.skip("Run src/data/preprocess.py first")
    return pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))


@pytest.fixture(scope="session")
def X_test():
    if not DATA_READY:
        pytest.skip("Run src/data/preprocess.py first")
    return pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))


@pytest.fixture(scope="session")
def y_train():
    if not DATA_READY:
        pytest.skip("Run src/data/preprocess.py first")
    return pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()


@pytest.fixture(scope="session")
def y_test():
    if not DATA_READY:
        pytest.skip("Run src/data/preprocess.py first")
    return pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()


@pytest.fixture(scope="session")
def pipeline():
    if not MODELS_READY:
        pytest.skip("Run src/models/train.py first")
    import joblib
    return joblib.load(os.path.join(MODELS_DIR, "pipeline.joblib"))


@pytest.fixture(scope="session")
def model():
    if not MODELS_READY:
        pytest.skip("Run src/models/train.py first")
    import joblib
    return joblib.load(os.path.join(MODELS_DIR, "model.joblib"))


SAMPLE_INPUT = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
}
