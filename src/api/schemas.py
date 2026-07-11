"""
schemas.py -- Pydantic request/response models for the Heart Disease Prediction API.
"""

from pydantic import BaseModel


class PatientFeatures(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


class PredictionResponse(BaseModel):
    prediction: int    # 0 = No Heart Disease, 1 = Heart Disease
    confidence: float  # probability of predicted class [0.0–1.0]
    label: str         # "Heart Disease" or "No Heart Disease"
