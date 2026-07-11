# Heart Disease MLOps

End-to-end MLOps pipeline for heart disease prediction — BITS Pilani AIMLCZG523 Assignment 01.

| | |
|---|---|
| **Name** | Sampath Kumar S Shetty |
| **Student ID** | 2024ac05041 |
| **Course** | AIMLCZG523 — Machine Learning Operations |
| **Institution** | BITS Pilani (M.Tech AI/ML) |
| **Documentation** | https://sampathshetty85.github.io/heart-disease-mlops/ |
| **Report (PDF)** | [docs/report.pdf](https://sampathshetty85.github.io/heart-disease-mlops/report.pdf) |

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

Everything runs inside Docker — no Python or pip setup needed on your machine. The pre-built image is publicly available on Docker Hub at [`sampathi348975/heart-disease-mlops`](https://hub.docker.com/r/sampathi348975/heart-disease-mlops).

### 1. Clone the repo
```bash
git clone https://github.com/sampathshetty85/heart-disease-mlops.git
cd heart-disease-mlops
```

### 2. Build the Docker image
```bash
docker build -t heart-disease-api .
```

This single command runs the full pipeline automatically, in order:
- Installs all Python dependencies
- Downloads the Heart Disease UCI dataset from UCI ML Repository
- Cleans and preprocesses the data
- Trains Logistic Regression, Random Forest, and XGBoost models
- Packages the best model (Logistic Regression, ROC-AUC 0.963) for serving

Takes 3–5 minutes on first build. Subsequent builds are faster due to Docker layer caching.

### 3. Run the container
```bash
docker run -p 8000:8000 heart-disease-api
```

The API starts immediately — the model is already baked into the image. No training happens at startup.

**Test it (open a new terminal):**
```bash
# Health check
curl http://localhost:8000/health

# Prediction — no heart disease likely
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}'

# Prediction — heart disease likely
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

Or open **http://localhost:8000/docs** for the interactive Swagger UI.

### Optional — Run with monitoring (Prometheus + Grafana)
```bash
docker-compose up -d
```

| Service | URL | Login |
|---------|-----|-------|
| API | http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

```bash
docker-compose down   # to stop
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

## Documentation

Full project documentation is available at:
**https://sampathshetty85.github.io/heart-disease-mlops/**

PDF Report: [docs/report.pdf](https://sampathshetty85.github.io/heart-disease-mlops/report.pdf)

## Video Demo

| # | Description | Link |
|---|---|---|
| 1 | End-to-end demo — API predictions, Kubernetes, Monitoring dashboards | [Watch](https://youtu.be/NYpTfF134Ug) |
| 2 | Docker Build — full pipeline runs automatically inside the image | [Watch](https://youtu.be/6ZizGNL5b-g) |
| 3 | Docker Run — container starts, API serves instantly | [Watch](https://youtu.be/8Ljn7DMnHX0) |
