# Heart Disease MLOps

End-to-end MLOps pipeline for heart disease prediction — BITS Pilani AIMLCZG523 Assignment 01.

| | |
|---|---|
| **Name** | Sampath Kumar S Shetty |
| **Student ID** | 2024ac05041 |
| **Course** | AIMLCZG523 — Machine Learning Operations |
| **Institution** | BITS Pilani (M.Tech AI/ML) |

## Problem Statement

Binary classifier predicting presence/absence of heart disease using the [Heart Disease UCI Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). Deployed as a cloud-ready, monitored REST API.

## Stack

| Layer | Tools |
|-------|-------|
| Data & ML | Pandas, NumPy, Scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Testing | Pytest |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Orchestration | Kubernetes (Minikube) |
| Monitoring | Prometheus + Grafana |

## Quick Start

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download dataset
```bash
python src/data/download.py
```

### 3. Preprocess data
```bash
python src/data/preprocess.py
```

### 4. Train models
```bash
python src/models/train.py
```

### 5. Launch MLflow UI
```bash
mlflow ui
# open http://localhost:5000
```

### 6. Run API locally
```bash
uvicorn src.api.main:app --reload --port 8000
# open http://localhost:8000/docs
```

### 7. Run tests
```bash
pytest tests/ -v
```

### 8. Build and run Docker container
```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

### 9. Deploy to Kubernetes (Minikube)
```bash
minikube start
kubectl apply -f k8s/
minikube service heart-disease-api --url
```

## Project Structure

```
heart-disease-mlops/
├── .github/workflows/    # CI/CD pipeline
├── data/                 # raw + processed datasets (gitignored)
├── notebooks/            # EDA and training notebooks
├── src/
│   ├── data/             # download + preprocess scripts
│   ├── models/           # training + evaluation
│   └── api/              # FastAPI app
├── tests/                # pytest unit tests
├── models/               # serialized artifacts (gitignored)
├── k8s/                  # Kubernetes manifests
├── monitoring/           # Prometheus + Grafana config
└── screenshots/          # submission evidence
```

## API Reference

### `GET /health`
Returns service health status.

### `POST /predict`
Accepts patient data, returns prediction and confidence score.

**Request body:**
```json
{
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 2,
  "thal": 3
}
```

**Response:**
```json
{
  "prediction": 0,
  "confidence": 0.87,
  "label": "No Heart Disease"
}
```

## Build Progress

See [build_plan.md](../build_plan.md) for task checklist and [build_log.md](../build_log.md) for phase-by-phase log.
