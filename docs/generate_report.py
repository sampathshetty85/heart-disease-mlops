"""
generate_report.py  --  Produces docs/report.pdf
10-page professional PDF report for AIMLCZG523 MLOps Assignment 01.
Run from repo root: python docs/generate_report.py
"""

import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.platypus.flowables import Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ---------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(REPO, "docs", "report.pdf")

W, H = A4
MARGIN = 2.0 * cm
MAX_IMG_W = 15.5 * cm  # gap 49 — max image width

# ---------------------------------------------------------------------------
# Colours
BG     = colors.HexColor("#0d1117")
ACCENT = colors.HexColor("#1f6feb")
GREEN  = colors.HexColor("#2ea043")
ORANGE = colors.HexColor("#e36209")
PURPLE = colors.HexColor("#9b59b6")
TEAL   = colors.HexColor("#0db7ed")
LGRAY  = colors.HexColor("#8b949e")
WHITE  = colors.HexColor("#e6edf3")
DKROW  = colors.HexColor("#161b22")
BORD   = colors.HexColor("#30363d")

base   = getSampleStyleSheet()

def style(name, parent="Normal", **kw):
    s = ParagraphStyle(name, parent=base[parent], **kw)
    return s

H1 = style("H1", fontSize=22, textColor=WHITE, leading=28,
           spaceAfter=6, spaceBefore=0, alignment=TA_LEFT,
           backColor=DKROW, leftIndent=-MARGIN, rightIndent=-MARGIN,
           borderPadding=(8, MARGIN, 8, MARGIN))
H2 = style("H2", fontSize=14, textColor=WHITE, leading=20,
           spaceAfter=4, spaceBefore=14, fontName="Helvetica-Bold")
H3 = style("H3", fontSize=11, textColor=colors.HexColor("#58a6ff"),
           leading=16, spaceAfter=2, spaceBefore=8, fontName="Helvetica-Bold")
BODY = style("BODY", fontSize=9, textColor=WHITE, leading=14,
             spaceAfter=4, backColor=None)
CODE = style("CODE", fontSize=8, textColor=colors.HexColor("#79c0ff"),
             fontName="Courier", leading=12, spaceAfter=2,
             backColor=DKROW, borderPadding=4, leftIndent=8)
SMALL = style("SMALL", fontSize=8, textColor=LGRAY, leading=12, spaceAfter=2)
CAPTION = style("CAPTION", fontSize=8, textColor=LGRAY, leading=11,
                alignment=TA_CENTER, spaceAfter=6)
COVER_TITLE = style("COVER_TITLE", fontSize=30, textColor=WHITE,
                    leading=38, alignment=TA_CENTER, fontName="Helvetica-Bold")
COVER_SUB = style("COVER_SUB", fontSize=13, textColor=LGRAY,
                  leading=18, alignment=TA_CENTER)
COVER_GREEN = style("COVER_GREEN", fontSize=11, textColor=GREEN,
                    leading=16, alignment=TA_CENTER, fontName="Helvetica-Bold")

def img(rel_path, width=None, caption=None):
    """Return [RLImage, Spacer] or [] if file missing."""
    path = os.path.join(REPO, rel_path)
    if not os.path.exists(path):
        return [Paragraph(f"[Image not found: {rel_path}]", SMALL)]
    w = width or MAX_IMG_W
    item = RLImage(path, width=w, height=w * 0.62, kind="proportional")
    result = [item]
    if caption:
        result.append(Paragraph(caption, CAPTION))
    result.append(Spacer(1, 4))
    return result

def hr():
    return HRFlowable(width="100%", thickness=1, color=BORD,
                      spaceAfter=6, spaceBefore=6)

def tbl(data, col_widths=None, header_bg=ACCENT):
    t = Table(data, colWidths=col_widths)
    n_rows = len(data)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [DKROW, BG]),
        ("TEXTCOLOR", (0, 1), (-1, -1), WHITE),
        ("GRID", (0, 0), (-1, -1), 0.5, BORD),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style_cmds))
    return t

# ---------------------------------------------------------------------------
# Page template with header/footer
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)

    # header bar
    canvas.setFillColor(DKROW)
    canvas.rect(0, H - 1.0*cm, W, 1.0*cm, fill=1, stroke=0)
    canvas.setFillColor(LGRAY)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(MARGIN, H - 0.65*cm,
                      "Heart Disease MLOps  ·  AIMLCZG523  ·  Sampath Kumar S Shetty (2024ac05041)")
    canvas.drawRightString(W - MARGIN, H - 0.65*cm,
                           "github.com/sampathshetty85/heart-disease-mlops")

    # footer
    canvas.setFillColor(DKROW)
    canvas.rect(0, 0, W, 0.9*cm, fill=1, stroke=0)
    canvas.setFillColor(LGRAY)
    canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(W / 2, 0.32*cm, f"Page {doc.page}")
    canvas.restoreState()

# ---------------------------------------------------------------------------
def build():
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.4*cm, bottomMargin=1.2*cm,
        title="Heart Disease MLOps — Assignment Report",
        author="Sampath Kumar S Shetty",
        subject="AIMLCZG523 MLOps Assignment 01",
    )

    story = []

    # ------------------------------------------------------------------
    # PAGE 1 — COVER
    # ------------------------------------------------------------------
    story += [
        Spacer(1, 2.5*cm),
        Paragraph("Heart Disease MLOps", COVER_TITLE),
        Spacer(1, 0.5*cm),
        Paragraph("End-to-End ML Pipeline — Binary Classification API", COVER_SUB),
        Spacer(1, 1.0*cm),
        hr(),
        Spacer(1, 0.5*cm),
        Paragraph("Machine Learning Operations (MLOps) — AIMLCZG523", COVER_SUB),
        Spacer(1, 0.3*cm),
        Paragraph("Assignment 01", COVER_SUB),
        Spacer(1, 1.0*cm),
        Paragraph("Student: Sampath Kumar S Shetty", COVER_GREEN),
        Paragraph("ID: 2024ac05041", COVER_GREEN),
        Spacer(1, 0.4*cm),
        Paragraph("GitHub: https://github.com/sampathshetty85/heart-disease-mlops", COVER_SUB),
        Paragraph("Pages: https://sampathshetty85.github.io/heart-disease-mlops/", COVER_SUB),
        Spacer(1, 1.0*cm),
        hr(),
        Spacer(1, 0.5*cm),
        tbl(
            [["Component", "Technology"],
             ["Language", "Python 3.12.2"],
             ["ML Framework", "scikit-learn 1.5.0 + XGBoost 2.0.3"],
             ["Experiment Tracking", "MLflow 2.22.5"],
             ["API Framework", "FastAPI 0.111.0 + Uvicorn"],
             ["Containerisation", "Docker (python:3.12-slim)"],
             ["Orchestration", "Kubernetes / Minikube v1.36.0"],
             ["Monitoring", "Prometheus v2.53.0 + Grafana 11.1.0"],
             ["CI/CD", "GitHub Actions (10-step pipeline)"],
             ["Testing", "Pytest (27 tests, 0 failures)"]],
            col_widths=[6*cm, 9.5*cm],
        ),
        PageBreak(),
    ]

    # ------------------------------------------------------------------
    # PAGE 2 — PROJECT OVERVIEW
    # ------------------------------------------------------------------
    story += [
        Paragraph("1. Project Overview", H1),
        Spacer(1, 0.3*cm),
        Paragraph(
            "This project implements a production-ready MLOps pipeline for heart disease risk "
            "prediction. Starting from raw UCI data, the pipeline covers data acquisition, "
            "preprocessing, model training, experiment tracking, packaging, CI/CD, "
            "containerisation, Kubernetes deployment, and Prometheus/Grafana monitoring — "
            "mirroring real-world production MLOps practices.", BODY),
        Spacer(1, 0.2*cm),
        Paragraph("Dataset", H3),
        tbl(
            [["Property", "Value"],
             ["Name", "Heart Disease UCI Dataset"],
             ["Source", "UCI Machine Learning Repository (ucimlrepo id=45)"],
             ["Rows", "303 patients"],
             ["Features", "13 (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)"],
             ["Target", "Binary — 0: No Heart Disease (164, 54.1%) / 1: Heart Disease (139, 45.9%)"],
             ["Missing values", "ca: 4 missing → median imputed; thal: 2 missing → median imputed"],
             ["Train / Test split", "80/20 stratified — 242 train / 61 test"]],
            col_widths=[4.5*cm, 11*cm],
        ),
        Spacer(1, 0.3*cm),
        Paragraph("Problem Statement", H3),
        Paragraph(
            "Build a binary classifier to predict presence or absence of heart disease from "
            "patient health data. Deploy the model as a cloud-ready, monitored REST API that "
            "accepts JSON input and returns a prediction with confidence score.", BODY),
        Spacer(1, 0.3*cm),
        Paragraph("Architecture Overview", H3),
    ] + img("screenshots/architecture_diagram.png",
            caption="Figure 1 — End-to-end system architecture: data pipeline → model training → CI/CD → Kubernetes → monitoring") + [
        PageBreak(),
    ]

    # ------------------------------------------------------------------
    # PAGE 3 — SETUP & INSTALLATION
    # ------------------------------------------------------------------
    story += [
        Paragraph("2. Setup &amp; Installation", H1),
        Spacer(1, 0.3*cm),
        Paragraph("Prerequisites", H3),
        Paragraph(
            "Docker Desktop is the only hard requirement. "
            "Python, pip, and all dependencies are handled inside the container.", BODY),
        Spacer(1, 0.3*cm),
        Paragraph("Quick Start — 3 Steps (Docker)", H3),
        Paragraph(
            "Clone the repository, build the image, and run the container. "
            "That is all. The build step runs the full pipeline automatically. "
            "The pre-built image is also publicly available on Docker Hub at "
            "sampathi348975/heart-disease-mlops.", BODY),
        Spacer(1, 0.25*cm),

        Paragraph("Step 1 — Clone the repository", H3),
        Paragraph("Download the project to your local machine.", BODY),
        Paragraph("git clone https://github.com/sampathshetty85/heart-disease-mlops.git", CODE),
        Paragraph("cd heart-disease-mlops", CODE),
        Spacer(1, 0.2*cm),

        Paragraph("Step 2 — Build the Docker image", H3),
        Paragraph(
            "This single command runs the full pipeline automatically — installs all dependencies, "
            "downloads the Heart Disease UCI dataset, cleans and preprocesses the data, trains "
            "Logistic Regression, Random Forest, and XGBoost models, then packages the best one. "
            "Takes 3–5 minutes on first run. Subsequent builds are faster due to Docker layer caching.", BODY),
        Paragraph("docker build -t heart-disease-api .", CODE),
        Spacer(1, 0.2*cm),

        Paragraph("Step 3 — Run the container", H3),
        Paragraph(
            "The API starts immediately. The model is already baked into the image — "
            "no training happens at startup.", BODY),
        Paragraph("docker run -p 8000:8000 heart-disease-api", CODE),
        Spacer(1, 0.2*cm),

        Paragraph("Test it (open a new terminal)", H3),
        Paragraph(
            "Run the commands below, or open http://localhost:8000/docs "
            "for the interactive Swagger UI.", BODY),
        Paragraph("curl http://localhost:8000/health", CODE),
        Paragraph('curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \\', CODE),
        Paragraph('  -d \'{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,', CODE),
        Paragraph('        "restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}\'', CODE),
        Spacer(1, 0.25*cm),

        Paragraph("Optional — Full Monitoring Stack (API + Prometheus + Grafana)", H3),
        Paragraph(
            "Starts all three services together. Grafana dashboard is pre-configured "
            "at localhost:3000 (login: admin / admin).", BODY),
        Paragraph("docker-compose up -d", CODE),
        Spacer(1, 0.2*cm),

        Paragraph("Optional — Kubernetes (Minikube)", H3),
        Paragraph("minikube start --driver=docker", CODE),
        Paragraph("kubectl apply -f k8s/deployment.yaml", CODE),
        Paragraph("kubectl apply -f k8s/service.yaml", CODE),
        Paragraph("kubectl port-forward svc/heart-disease-svc 8080:80", CODE),
        Spacer(1, 0.3*cm),

        Paragraph("CI/CD", H3),
        Paragraph(
            "Every push to main triggers GitHub Actions (ci.yml). "
            "The 10-step pipeline runs: flake8 lint → download → preprocess → train → "
            "pytest (27 tests) → upload-artifact. Pipeline fails loudly on any error.", BODY),
        PageBreak(),
    ]

    # ------------------------------------------------------------------
    # PAGE 4 — EDA
    # ------------------------------------------------------------------
    story += [
        Paragraph("3. Exploratory Data Analysis", H1),
        Spacer(1, 0.2*cm),
        Paragraph("Key Findings", H3),
        tbl(
            [["Feature", "Type", "Key Insight"],
             ["age", "Numeric", "Mean 54.4 yrs; disease patients slightly older"],
             ["thal", "Categorical", "Highest correlation with target (0.522)"],
             ["ca", "Categorical", "2nd highest correlation (0.460)"],
             ["exang", "Categorical", "Exercise-induced angina — strong predictor (0.432)"],
             ["cp (chest pain)", "Categorical", "Type 0 (typical angina) strongly associated with disease"],
             ["chol", "Numeric", "Mean 246.7; wide range 126–564; weak direct correlation"],
             ["oldpeak", "Numeric", "ST depression — moderate correlation with disease"]],
            col_widths=[3.0*cm, 2.8*cm, 9.7*cm],
        ),
        Spacer(1, 0.2*cm),
        Paragraph("Class Balance", H3),
        Paragraph(
            "164 patients without heart disease (54.1%) vs 139 with heart disease (45.9%). "
            "Well-balanced — no resampling (SMOTE/undersampling) required. "
            "Stratified 80/20 split preserves this ratio in both train and test sets.", BODY),
        Spacer(1, 0.2*cm),
        Paragraph("Missing Values", H3),
        Paragraph(
            "Only ca (4 missing) and thal (2 missing) had null values. "
            "Both imputed with median and cast to int — required to prevent "
            "OneHotEncoder float/int category mismatch at inference time.", BODY),
        Spacer(1, 0.2*cm),
    ] + img("screenshots/eda/correlation_heatmap.png", width=12*cm,
            caption="Figure 2 — Feature correlation heatmap: thal (0.522), ca (0.460), exang (0.432) top predictors") + [
        Spacer(1, 0.1*cm),
    ] + img("screenshots/eda/class_distribution.png", width=9*cm,
            caption="Figure 3 — Class distribution: 54.1% no-disease / 45.9% disease (well-balanced)") + [
        PageBreak(),
    ]

    # ------------------------------------------------------------------
    # PAGE 5 — MODEL DEVELOPMENT
    # ------------------------------------------------------------------
    story += [
        Paragraph("4. Feature Engineering &amp; Model Development", H1),
        Spacer(1, 0.2*cm),
        Paragraph("Preprocessing Pipeline", H3),
        Paragraph(
            "A single sklearn ColumnTransformer is used inside a Pipeline — "
            "ensuring the scaler and encoder are fit only on training data, "
            "preventing data leakage. The same pipeline is serialised to "
            "models/pipeline.joblib and used at inference time.", BODY),
        tbl(
            [["Step", "Transformer", "Features"],
             ["Scaling", "StandardScaler", "age, trestbps, chol, thalach, oldpeak (5 numeric)"],
             ["Encoding", "OneHotEncoder(drop='first', handle_unknown='ignore')",
              "sex, cp, fbs, restecg, exang, slope, ca, thal (8 categorical)"]],
            col_widths=[2.5*cm, 5.5*cm, 7.5*cm],
        ),
        Spacer(1, 0.2*cm),
        Paragraph("Hyperparameter Tuning", H3),
        tbl(
            [["Model", "Search Strategy", "Key Hyperparameters Tuned"],
             ["Logistic Regression", "GridSearchCV (5-fold)", "C ∈ {0.01,0.1,1,10,100}, solver ∈ {liblinear,lbfgs}"],
             ["Random Forest", "RandomizedSearchCV (5-fold, 20 iter)", "n_estimators, max_depth, min_samples_split, max_features"],
             ["XGBoost", "RandomizedSearchCV (5-fold, 20 iter)", "n_estimators, max_depth, learning_rate, subsample, colsample_bytree"]],
            col_widths=[3.5*cm, 4.5*cm, 7.5*cm],
        ),
        Spacer(1, 0.2*cm),
        Paragraph("Model Comparison — 5-fold Stratified Cross-Validation on Training Set", H3),
        tbl(
            [["Model", "CV ROC-AUC", "CV F1", "CV Accuracy"],
             ["Logistic Regression ✓ BEST", "0.9026 ± 0.0162", "0.8252 ± 0.0171", "0.8471 ± 0.0208"],
             ["Random Forest", "0.8845 ± 0.0270", "0.7614 ± 0.0221", "0.7933 ± 0.0233"],
             ["XGBoost", "0.8690 ± 0.0188", "0.7498 ± 0.0247", "0.7770 ± 0.0222"]],
            col_widths=[5.0*cm, 4.0*cm, 3.5*cm, 3.0*cm],
            header_bg=GREEN,
        ),
        Spacer(1, 0.2*cm),
        Paragraph("Model Comparison — Test Set (n=61)", H3),
        tbl(
            [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
             ["Logistic Regression ✓ BEST", "0.8689", "0.8333", "0.8929", "0.8621", "0.9632"],
             ["Random Forest", "0.8689", "0.8333", "0.8929", "0.8621", "0.9481"],
             ["XGBoost", "0.8852", "0.8621", "0.8929", "0.8772", "0.9383"]],
            col_widths=[5.0*cm, 2.3*cm, 2.3*cm, 2.3*cm, 2.0*cm, 1.6*cm],
            header_bg=GREEN,
        ),
        Spacer(1, 0.2*cm),
        Paragraph("Selection Rationale", H3),
        Paragraph(
            "Logistic Regression selected as the best model based on highest 5-fold CV ROC-AUC "
            "(0.9026). While XGBoost achieved marginally higher test accuracy (0.8852 vs 0.8689), "
            "CV ROC-AUC is the primary criterion for generalisation. "
            "Best hyperparameters: C=10, solver=liblinear.", BODY),
        Spacer(1, 0.1*cm),
    ] + img("screenshots/eda/roc_curves.png", width=12*cm,
            caption="Figure 4 — ROC curves for all 3 models on test set (LR AUC=0.9632)") + [
        PageBreak(),
    ]

    # ------------------------------------------------------------------
    # PAGE 6 — EXPERIMENT TRACKING
    # ------------------------------------------------------------------
    story += [
        Paragraph("5. Experiment Tracking — MLflow", H1),
        Spacer(1, 0.2*cm),
        Paragraph("Configuration", H3),
        tbl(
            [["Property", "Value"],
             ["Experiment name", "heart-disease-mlops"],
             ["Tracking URI", "mlruns/ (anchored to REPO_ROOT, Docker-safe)"],
             ["MLflow version", "2.22.5"],
             ["Total runs", "3 (one per model)"],
             ["Best run", "logistic_regression (tag: best_model=true)"]],
            col_widths=[4.5*cm, 11*cm],
        ),
        Spacer(1, 0.2*cm),
        Paragraph("Logged Per Run", H3),
        tbl(
            [["Category", "Items Logged"],
             ["Parameters (2)", "classifier__C, classifier__solver (best hyperparams)"],
             ["Metrics (10)", "cv_roc_auc, cv_roc_auc_std, cv_f1, cv_f1_std, cv_accuracy, test_accuracy, test_precision, test_recall, test_f1, test_roc_auc"],
             ["Artifacts (3)", "roc_curves.png, confusion_matrices.png, full sklearn Pipeline model"],
             ["Model", "mlflow.sklearn.log_model() with input_example + infer_signature()"],
             ["Tags", "best_model=true on Logistic Regression run"]],
            col_widths=[3.5*cm, 12.0*cm],
        ),
        Spacer(1, 0.2*cm),
        Paragraph("MLflow Run IDs", H3),
        tbl(
            [["Model", "Run ID", "Best?"],
             ["Logistic Regression", "e4abcc6349dd4254ac77172fbd19d08c", "✓ Yes"],
             ["Random Forest", "3f3b2fc0795b48f886ef10886275a559", "No"],
             ["XGBoost", "ec91c4a1867644a7b1abba1e74744513", "No"]],
            col_widths=[3.5*cm, 9.5*cm, 2.5*cm],
        ),
        Spacer(1, 0.15*cm),
        Paragraph("Model URI: runs:/e4abcc6349dd4254ac77172fbd19d08c/model", CODE),
        Spacer(1, 0.2*cm),
    ] + img("screenshots/mlflow/experiment_runs.png",
            caption="Figure 5 — MLflow experiment UI: 3 runs with parameters, metrics, and artifacts") + [
        PageBreak(),
    ]

    # ------------------------------------------------------------------
    # PAGE 7 — CI/CD & CONTAINERISATION
    # ------------------------------------------------------------------
    story += [
        Paragraph("6. CI/CD Pipeline &amp; Automated Testing", H1),
        Spacer(1, 0.2*cm),
        Paragraph("GitHub Actions — 10-Step Pipeline (.github/workflows/ci.yml)", H3),
        tbl(
            [["Step", "Action", "Purpose"],
             ["1", "actions/checkout@v4", "Clone repository"],
             ["2", "actions/setup-python@v5 (3.12)", "Set up Python environment"],
             ["3", "actions/cache@v4", "Cache pip dependencies"],
             ["4", "pip install -r requirements.txt", "Install all 19 dependencies"],
             ["5", "flake8 src/ tests/ --max-line-length=120", "Lint — fails on any violation"],
             ["6", "python src/data/download.py", "Fetch live UCI dataset"],
             ["7", "python src/data/preprocess.py", "Clean and split data"],
             ["8", "python src/models/train.py", "Train all 3 models"],
             ["9", "pytest tests/ -v", "Run 27 unit tests — fails on any failure"],
             ["10", "actions/upload-artifact@v4", "Upload pipeline output reports"]],
            col_widths=[1.0*cm, 6.0*cm, 8.5*cm],
        ),
        Spacer(1, 0.15*cm),
        Paragraph("Test Suite (27 tests, 0 failures, 0 skipped)", H3),
        tbl(
            [["File", "Tests", "Covers"],
             ["test_preprocess.py", "8", "load_raw, handle_missing, binarize_target, split, save_splits"],
             ["test_model.py", "8", "build_pipeline, train, evaluate, best_model selection, artifact save"],
             ["test_predict.py", "5", "load_artifacts, predict, confidence bounds, label strings"],
             ["test_api.py", "6", "GET /health, POST /predict valid/invalid, GET /metrics, 422 handling"]],
            col_widths=[4.0*cm, 1.8*cm, 9.7*cm],
        ),
        Spacer(1, 0.2*cm),
    ] + img("screenshots/ci/ci_passing.png",
            caption="Figure 6 — GitHub Actions: all 10 steps passing (Run #29148726844, 1m 38s)") + [
        Spacer(1, 0.2*cm),
        Paragraph("7. Model Containerisation", H1),
        Spacer(1, 0.2*cm),
        Paragraph("Dockerfile Strategy", H3),
        Paragraph(
            "python:3.12-slim base. Four RUN build steps at docker build time fetch live UCI data, "
            "preprocess, train, and verify artifacts. Model is baked into the image layer — "
            "container startup is instant (1094ms restart confirmed, no retraining).", BODY),
    ] + img("screenshots/docker/swagger_ui.png", width=12*cm,
            caption="Figure 7 — FastAPI Swagger UI: /health, /predict, /metrics endpoints") + [
        PageBreak(),
    ]

    # ------------------------------------------------------------------
    # PAGE 8 — KUBERNETES
    # ------------------------------------------------------------------
    story += [
        Paragraph("8. Production Deployment — Kubernetes", H1),
        Spacer(1, 0.2*cm),
        Paragraph("Cluster Configuration", H3),
        tbl(
            [["Property", "Value"],
             ["Platform", "Minikube v1.36.0, Kubernetes v1.33.1, docker driver"],
             ["Image", "sampathi348975/heart-disease-mlops:latest (Docker Hub)"],
             ["Replicas", "2 (Deployment spec)"],
             ["imagePullPolicy", "Always (guarantees fresh pull from registry)"],
             ["Liveness probe", "GET /health, initialDelaySeconds=5, periodSeconds=10"],
             ["Readiness probe", "GET /health, initialDelaySeconds=5, periodSeconds=5"],
             ["Resources", "requests: 256Mi/250m  limits: 512Mi/500m"],
             ["Service type", "LoadBalancer (port 80 → containerPort 8000)"],
             ["Access method", "kubectl port-forward svc/heart-disease-svc 8080:80"]],
            col_widths=[4.5*cm, 11*cm],
        ),
        Spacer(1, 0.2*cm),
        Paragraph("Model Refresh Procedure", H3),
        Paragraph("1. docker build -t sampathi348975/heart-disease-mlops:latest .", CODE),
        Paragraph("2. docker push sampathi348975/heart-disease-mlops:latest", CODE),
        Paragraph("3. kubectl rollout restart deployment/heart-disease-api", CODE),
        Spacer(1, 0.2*cm),
    ] + img("screenshots/k8s/pods_running.png", width=12*cm,
            caption="Figure 8 — kubectl get pods: 2/2 Running, READY, STATUS=Running") + [
        Spacer(1, 0.1*cm),
    ] + img("screenshots/k8s/predict_response.png", width=12*cm,
            caption="Figure 9 — POST /predict via K8s port-forward: prediction=0, confidence=0.9357") + [
        PageBreak(),
    ]

    # ------------------------------------------------------------------
    # PAGE 9 — MONITORING
    # ------------------------------------------------------------------
    story += [
        Paragraph("9. Monitoring &amp; Logging", H1),
        Spacer(1, 0.2*cm),
        Paragraph("API Request Logging", H3),
        Paragraph(
            "Every POST /predict call emits a structured JSON log entry via Python logging. "
            "Startup event is also logged with model name and timestamp.", BODY),
        Paragraph(
            '{"timestamp": "2026-07-11T09:58:02Z", "endpoint": "/predict", '
            '"prediction": 0, "confidence": 0.9213, "label": "No Heart Disease", "latency_ms": 3.12}',
            CODE),
        Spacer(1, 0.2*cm),
        Paragraph("Prometheus Metrics (prometheus-fastapi-instrumentator)", H3),
        tbl(
            [["Metric", "Type", "Description"],
             ["http_requests_total", "Counter", "Request count by handler, method, status code"],
             ["http_request_duration_seconds", "Histogram", "Request latency buckets (p50, p95 queryable)"],
             ["heart_disease_predictions_total", "Counter", "Prediction count by outcome label"],
             ["python_info", "Gauge", "Python runtime version info"]],
            col_widths=[5.5*cm, 2.5*cm, 7.5*cm],
        ),
        Spacer(1, 0.15*cm),
        Paragraph("Prometheus metrics sample (from /metrics endpoint):", H3),
        Paragraph("http_requests_total{handler='/predict',method='POST',status='2xx'} 20.0", CODE),
        Paragraph("http_requests_total{handler='/predict',method='POST',status='4xx'} 2.0", CODE),
        Paragraph("http_request_duration_highr_seconds_bucket{le='0.01'} 26.0", CODE),
        Spacer(1, 0.2*cm),
        Paragraph("Grafana Dashboard — 4 Panels", H3),
        tbl(
            [["Panel", "PromQL Query"],
             ["API Request Rate (req/s)", "rate(http_requests_total[1m])"],
             ["Latency p50 / p95", "histogram_quantile(0.50/0.95, rate(http_request_duration_seconds_bucket[5m]))"],
             ["Prediction Distribution", "increase(heart_disease_predictions_total[5m])"],
             ["Error Rate (4xx/5xx)", "rate(http_requests_total{status_code=~'4..|5..'}[1m])"]],
            col_widths=[5.5*cm, 10.0*cm],
        ),
        Spacer(1, 0.15*cm),
    ] + img("screenshots/monitoring/grafana_dashboard.png",
            caption="Figure 10 — Grafana dashboard: request rate, latency p50/p95, prediction distribution, error rate") + [
        PageBreak(),
    ]

    # ------------------------------------------------------------------
    # PAGE 10 — REPOSITORY & ACCESS
    # ------------------------------------------------------------------
    story += [
        Paragraph("10. Repository, Access &amp; Deliverables", H1),
        Spacer(1, 0.2*cm),
        Paragraph("GitHub Repository", H3),
        Paragraph("https://github.com/sampathshetty85/heart-disease-mlops", CODE),
        Spacer(1, 0.15*cm),
        Paragraph("GitHub Pages Documentation", H3),
        Paragraph("https://sampathshetty85.github.io/heart-disease-mlops/", CODE),
        Spacer(1, 0.2*cm),
        Paragraph("Repository Structure", H3),
        tbl(
            [["Path", "Contents"],
             ["src/data/", "download.py, preprocess.py"],
             ["src/models/", "train.py, predict.py, package.py"],
             ["src/api/", "main.py, schemas.py"],
             ["notebooks/", "01_eda.ipynb (17 cells, fully executed)"],
             ["tests/", "27 pytest tests across 4 files"],
             ["k8s/", "deployment.yaml, service.yaml"],
             ["monitoring/", "prometheus.yml, grafana provisioning, dashboard JSON"],
             ["docs/", "report.pdf, generate_report.py, index.html, style.css"],
             ["output/", "10 timestamped validation reports (all phases)"],
             ["screenshots/", "17 screenshots across ci/, docker/, eda/, k8s/, mlflow/, monitoring/"],
             [".github/workflows/", "ci.yml — 10-step GitHub Actions pipeline"]],
            col_widths=[4.5*cm, 11.0*cm],
        ),
        Spacer(1, 0.3*cm),
        Paragraph("Local Access Instructions", H3),
        tbl(
            [["Service", "Command", "URL"],
             ["API (local)", "uvicorn src.api.main:app --port 8000", "http://localhost:8000"],
             ["API (Docker)", "docker run -p 8000:8000 sampathi348975/heart-disease-mlops:latest", "http://localhost:8000"],
             ["API (K8s)", "kubectl port-forward svc/heart-disease-svc 8080:80", "http://localhost:8080"],
             ["Monitoring stack", "docker-compose up -d (from repo root)", "localhost:3000 (Grafana)"],
             ["MLflow UI", "mlflow ui --backend-store-uri mlruns/", "http://localhost:5000"],
             ["Swagger UI", "— (built into FastAPI)", "http://localhost:8000/docs"]],
            col_widths=[3.0*cm, 7.0*cm, 5.5*cm],
        ),
        Spacer(1, 0.3*cm),
        Paragraph("Video Demo", H3),
        tbl(
            [["#", "Description", "URL"],
             ["1", "End-to-end demo — API predictions, Kubernetes, Monitoring dashboards", "https://youtu.be/NYpTfF134Ug"],
             ["2", "Docker Build — full pipeline runs automatically inside the image", "https://youtu.be/6ZizGNL5b-g"],
             ["3", "Docker Run — container starts, API serves instantly", "https://youtu.be/8Ljn7DMnHX0"]],
            col_widths=[0.8*cm, 8.5*cm, 6.2*cm],
        ),
        Spacer(1, 0.3*cm),
        hr(),
        Spacer(1, 0.3*cm),
        Paragraph("Deliverables Checklist", H3),
        tbl(
            [["Deliverable", "Status", "Location"],
             ["GitHub repository (public)", "✓", "github.com/sampathshetty85/heart-disease-mlops"],
             ["requirements.txt (19 pinned deps)", "✓", "requirements.txt"],
             ["Download script", "✓", "src/data/download.py"],
             ["EDA notebook (17 cells)", "✓", "notebooks/01_eda.ipynb"],
             ["Pytest suite (27 tests)", "✓", "tests/"],
             ["GitHub Actions workflow", "✓", ".github/workflows/ci.yml"],
             ["Dockerfile", "✓", "Dockerfile"],
             ["Docker Compose (monitoring)", "✓", "docker-compose.yml"],
             ["K8s manifests", "✓", "k8s/deployment.yaml, k8s/service.yaml"],
             ["Screenshot folder (17 images)", "✓", "screenshots/"],
             ["10-page PDF report", "✓", "docs/report.pdf"],
             ["GitHub Pages site", "✓", "sampathshetty85.github.io/heart-disease-mlops/"],
             ["Output validation reports (10)", "✓", "output/"]],
            col_widths=[6.5*cm, 1.5*cm, 7.5*cm],
        ),
    ]

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    size = os.path.getsize(OUT)
    print(f"Report generated: {OUT} ({size:,} bytes, ~{size//1024}KB)")


if __name__ == "__main__":
    build()
