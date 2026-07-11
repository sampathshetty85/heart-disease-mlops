"""
predict.py -- standalone inference module (Phase 2 deliverable).

Loads the serialized pipeline (models/pipeline.joblib) and returns
a prediction + confidence for a single patient record supplied as a dict.

Used by:
  - src/api/main.py  (Phase 6 FastAPI /predict endpoint)
  - Ad-hoc inference from the CLI

Note: models/pipeline.joblib is generated at docker build time by train.py.
      feature_columns.json (also build-time) governs column ordering.
"""

import os
import sys
import json
import pandas as pd
import joblib

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT     = os.path.join(BASE_DIR, "..", "..")
PIPELINE_PATH = os.path.join(REPO_ROOT, "models", "pipeline.joblib")
FEAT_COLS_PATH = os.path.join(REPO_ROOT, "data", "processed", "feature_columns.json")

_pipeline = None
_feature_columns = None


def load_artifacts():
    global _pipeline, _feature_columns
    if _pipeline is None:
        if not os.path.exists(PIPELINE_PATH):
            raise FileNotFoundError(
                "models/pipeline.joblib not found. Run src/models/train.py first."
            )
        _pipeline = joblib.load(PIPELINE_PATH)

    if _feature_columns is None:
        if not os.path.exists(FEAT_COLS_PATH):
            raise FileNotFoundError(
                "data/processed/feature_columns.json not found. Run src/data/preprocess.py first."
            )
        with open(FEAT_COLS_PATH) as f:
            _feature_columns = json.load(f)

    return _pipeline, _feature_columns


def predict(input_dict: dict) -> dict:
    """
    Accept a raw patient feature dict and return prediction + confidence.

    Args:
        input_dict: keys must match feature_columns.json (13 features)

    Returns:
        {"prediction": int, "confidence": float, "label": str}
    """
    pipeline, feature_columns = load_artifacts()

    df = pd.DataFrame([input_dict])[feature_columns]

    prediction  = int(pipeline.predict(df)[0])
    confidence  = float(pipeline.predict_proba(df)[0][prediction])
    label       = "Heart Disease" if prediction == 1 else "No Heart Disease"

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "label":      label,
    }


if __name__ == "__main__":
    # Smoke test with a sample patient record
    sample = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }
    result = predict(sample)
    print(f"Prediction : {result['prediction']}  ({result['label']})")
    print(f"Confidence : {result['confidence']:.4f}")
