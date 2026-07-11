"""
train.py -- Phase 2 & 3: Feature Engineering, Model Development & MLflow Tracking.

PIPELINE DESIGN
- ColumnTransformer wraps two transformers applied to feature subsets:
    StandardScaler    -> NUMERIC_FEATURES  (age, trestbps, chol, thalach, oldpeak)
    OneHotEncoder     -> CATEGORICAL_FEATURES
        drop='first'          : removes one dummy per feature to prevent multicollinearity
                                in Logistic Regression; harmless for tree-based models.
        handle_unknown='ignore': silently zero-fills unseen categories at API inference
                                 so the serving pipeline never crashes on novel inputs.
- A single shared preprocessor is used for all three models.
- Each final estimator is wrapped as Pipeline([preprocessor, classifier]) to ensure
  the transformer is fit only on X_train (data leakage prevention).

MODEL SELECTION CRITERION
- Primary  : highest mean ROC-AUC across 5-fold stratified CV on X_train
- Tiebreaker: highest mean F1-score (macro) across the same CV folds
- Rationale: ROC-AUC measures discrimination ability independent of threshold;
             F1 captures precision-recall balance for the imbalanced-adjacent case.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.models

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from report_writer import write_report

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.join(BASE_DIR, "..", "..")
DATA_DIR   = os.path.join(REPO_ROOT, "data", "processed")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
SHOTS_DIR  = os.path.join(REPO_ROOT, "screenshots", "eda")
MLFLOW_DIR = os.path.join(REPO_ROOT, "mlruns")
MLFLOW_EXPERIMENT = "heart-disease-mlops"

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ---------------------------------------------------------------------------
# A -- Load processed splits
# ---------------------------------------------------------------------------
def load_splits():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

    with open(os.path.join(DATA_DIR, "feature_columns.json")) as f:
        expected_cols = json.load(f)
    assert list(X_train.columns) == expected_cols, \
        "X_train column order does not match feature_columns.json"

    print(f"Loaded: X_train {X_train.shape}  X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# B -- Build preprocessing pipeline
# ---------------------------------------------------------------------------
def build_preprocessor():
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(classifier):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])


# ---------------------------------------------------------------------------
# C -- Hyperparameter tuning
# ---------------------------------------------------------------------------
def tune_logistic(X_train, y_train):
    print("\n--- Logistic Regression (GridSearchCV) ---")
    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    param_grid = {
        "classifier__C":      [0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["lbfgs", "liblinear"],
    }
    search = GridSearchCV(pipe, param_grid, cv=CV, scoring="roc_auc", n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    print(f"Best params : {search.best_params_}")
    print(f"Best CV AUC : {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_random_forest(X_train, y_train):
    print("\n--- Random Forest (RandomizedSearchCV) ---")
    pipe = build_pipeline(RandomForestClassifier(random_state=42))
    param_dist = {
        "classifier__n_estimators":   [100, 200, 300],
        "classifier__max_depth":      [None, 5, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__max_features":   ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=20, cv=CV,
        scoring="roc_auc", random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"Best params : {search.best_params_}")
    print(f"Best CV AUC : {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_xgboost(X_train, y_train):
    print("\n--- XGBoost (RandomizedSearchCV) ---")
    pipe = build_pipeline(xgb.XGBClassifier(
        eval_metric="logloss", random_state=42, verbosity=0
    ))
    param_dist = {
        "classifier__n_estimators":  [100, 200, 300],
        "classifier__max_depth":     [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample":     [0.7, 0.8, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
    }
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=20, cv=CV,
        scoring="roc_auc", random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"Best params : {search.best_params_}")
    print(f"Best CV AUC : {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_


# ---------------------------------------------------------------------------
# D -- Cross-validation scores for a fitted pipeline
# ---------------------------------------------------------------------------
def cv_scores(pipeline, X_train, y_train, label):
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"}
    results = cross_validate(pipeline, X_train, y_train, cv=CV, scoring=scoring)
    out = {
        "roc_auc_mean": results["test_roc_auc"].mean(),
        "roc_auc_std":  results["test_roc_auc"].std(),
        "f1_mean":      results["test_f1"].mean(),
        "f1_std":       results["test_f1"].std(),
        "acc_mean":     results["test_accuracy"].mean(),
        "acc_std":      results["test_accuracy"].std(),
    }
    print(f"{label}: CV ROC-AUC = {out['roc_auc_mean']:.4f} ± {out['roc_auc_std']:.4f}  "
          f"F1 = {out['f1_mean']:.4f} ± {out['f1_std']:.4f}")
    return out


# ---------------------------------------------------------------------------
# E -- Test-set evaluation
# ---------------------------------------------------------------------------
def evaluate_on_test(pipeline, X_test, y_test, label):
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }
    print(f"{label}: acc={metrics['accuracy']:.4f}  prec={metrics['precision']:.4f}  "
          f"rec={metrics['recall']:.4f}  f1={metrics['f1']:.4f}  auc={metrics['roc_auc']:.4f}")
    return metrics, y_pred, y_proba


# ---------------------------------------------------------------------------
# F -- Plots
# ---------------------------------------------------------------------------
def plot_roc_curves(models_data, X_test, y_test):
    """models_data: list of (label, pipeline, color)"""
    import seaborn as sns
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, pipeline, color in models_data:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(SHOTS_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_confusion_matrices(models_data, X_test, y_test):
    import seaborn as sns
    sns.set_style("whitegrid")

    n = len(models_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, pipeline, _) in zip(axes, models_data):
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
            ax=ax, linewidths=0.5
        )
        ax.set_title(f"{label}\nConfusion Matrix", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

    fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(SHOTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# G -- Save artifacts
# ---------------------------------------------------------------------------
def save_artifacts(best_pipeline):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path    = os.path.join(MODELS_DIR, "model.joblib")
    pipeline_path = os.path.join(MODELS_DIR, "pipeline.joblib")
    joblib.dump(best_pipeline.named_steps["classifier"], model_path)
    joblib.dump(best_pipeline, pipeline_path)
    model_size    = os.path.getsize(model_path)
    pipeline_size = os.path.getsize(pipeline_path)
    print(f"Saved: models/model.joblib    ({model_size:,} bytes)")
    print(f"Saved: models/pipeline.joblib ({pipeline_size:,} bytes)")
    return model_path, pipeline_path, model_size, pipeline_size


# ---------------------------------------------------------------------------
# H -- Output report
# ---------------------------------------------------------------------------
def _write_report(results, best_name, best_params, cv_map, model_size, pipeline_size):
    lines = [
        "STEP 2 -- Feature Engineering & Model Development",
        "",
        "--- Preprocessing Pipeline ---",
        "ColumnTransformer:",
        f"  StandardScaler      -> {NUMERIC_FEATURES}",
        f"  OneHotEncoder       -> {CATEGORICAL_FEATURES}",
        "  drop='first'        -> prevents LR multicollinearity; harmless for trees",
        "  handle_unknown='ignore' -> safe at API inference for unseen categories",
        "",
        "--- Cross-Validation Results (5-fold stratified, on X_train) ---",
    ]
    for name, cv in cv_map.items():
        lines.append(
            f"  {name:<20}: ROC-AUC = {cv['roc_auc_mean']:.4f} +/- {cv['roc_auc_std']:.4f}  "
            f"F1 = {cv['f1_mean']:.4f} +/- {cv['f1_std']:.4f}  "
            f"Acc = {cv['acc_mean']:.4f} +/- {cv['acc_std']:.4f}"
        )

    lines += [
        "",
        "--- Test-Set Evaluation (held-out X_test, n=61) ---",
        f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}",
        "-" * 72,
    ]
    for name, m in results.items():
        lines.append(
            f"{name:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>9.4f}"
        )

    best_cv_auc = cv_map[best_name]["roc_auc_mean"]
    lines += [
        "",
        "--- Model Selection ---",
        f"Criterion      : highest CV ROC-AUC (primary); F1 tiebreaker",
        f"Best model     : {best_name}",
        f"CV ROC-AUC     : {best_cv_auc:.4f}",
        f"Best params    : {best_params}",
        "",
        "--- Saved Artifacts ---",
        f"models/model.joblib    : {model_size:,} bytes",
        f"models/pipeline.joblib : {pipeline_size:,} bytes",
        "Serialization format   : joblib",
        "Note: models/ is gitignored -- artifacts baked into Docker image at build time",
        "",
        "--- Charts ---",
        "screenshots/eda/roc_curves.png         : ROC curves all models overlaid",
        "screenshots/eda/confusion_matrices.png : confusion matrices all models",
        "",
        "Status : COMPLETE",
    ]
    write_report("step_2_train", lines)


# ---------------------------------------------------------------------------
# I -- MLflow run logger (Phase 3)
# ---------------------------------------------------------------------------
def log_run(run_name, pipeline, params, cv, test_metrics,
            artifact_paths, X_train, is_best):
    """Log one model's full results as an MLflow run. Returns the run_id."""
    with mlflow.start_run(run_name=run_name) as run:
        # Strip 'classifier__' prefix so param names are readable in the UI
        clean_params = {k.replace("classifier__", ""): v for k, v in params.items()}
        mlflow.log_params(clean_params)

        # CV metrics
        mlflow.log_metric("cv_roc_auc_mean", round(cv["roc_auc_mean"], 6))
        mlflow.log_metric("cv_roc_auc_std",  round(cv["roc_auc_std"],  6))
        mlflow.log_metric("cv_f1_mean",       round(cv["f1_mean"],      6))
        mlflow.log_metric("cv_f1_std",        round(cv["f1_std"],       6))
        mlflow.log_metric("cv_acc_mean",      round(cv["acc_mean"],     6))

        # Test-set metrics
        mlflow.log_metric("test_accuracy",   round(test_metrics["accuracy"],  6))
        mlflow.log_metric("test_precision",  round(test_metrics["precision"], 6))
        mlflow.log_metric("test_recall",     round(test_metrics["recall"],    6))
        mlflow.log_metric("test_f1",         round(test_metrics["f1"],        6))
        mlflow.log_metric("test_roc_auc",    round(test_metrics["roc_auc"],   6))

        # Plot artifacts
        for path in artifact_paths:
            if os.path.exists(path):
                mlflow.log_artifact(path)

        # Model with input schema (Gap 31)
        signature = mlflow.models.infer_signature(
            X_train, pipeline.predict(X_train)
        )
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            input_example=X_train.iloc[:5],
            signature=signature,
        )

        # Tags
        mlflow.set_tag("model_name",  run_name)
        mlflow.set_tag("best_model",  "true" if is_best else "false")

        run_id = run.info.run_id
        print(f"  MLflow run logged: {run_name}  run_id={run_id[:8]}...  best={is_best}")
        return run_id


# ---------------------------------------------------------------------------
# J -- MLflow output report (Phase 3)
# ---------------------------------------------------------------------------
def _write_mlflow_report(run_ids, cv_map, results, best_name):
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    exp_id = experiment.experiment_id if experiment else "unknown"

    lines = [
        "STEP 3 -- MLflow Experiment Tracking",
        "",
        f"Experiment name : {MLFLOW_EXPERIMENT}",
        f"Experiment ID   : {exp_id}",
        f"Tracking URI    : {mlflow.get_tracking_uri()}",
        "",
        "--- Runs ---",
    ]

    run_key_map = {
        "Logistic Regression": "logistic_regression",
        "Random Forest":       "random_forest",
        "XGBoost":             "xgboost",
    }

    for display_name, run_key in run_key_map.items():
        run_id  = run_ids.get(run_key, "unknown")
        cv      = cv_map[display_name]
        metrics = results[display_name]
        is_best = display_name == best_name
        lines += [
            "",
            f"Run: {run_key}",
            f"  run_id      : {run_id}",
            f"  best_model  : {'true' if is_best else 'false'}",
            f"  cv_roc_auc  : {cv['roc_auc_mean']:.4f} +/- {cv['roc_auc_std']:.4f}",
            f"  cv_f1       : {cv['f1_mean']:.4f} +/- {cv['f1_std']:.4f}",
            f"  test_acc    : {metrics['accuracy']:.4f}",
            f"  test_prec   : {metrics['precision']:.4f}",
            f"  test_recall : {metrics['recall']:.4f}",
            f"  test_f1     : {metrics['f1']:.4f}",
            f"  test_auc    : {metrics['roc_auc']:.4f}",
            f"  artifacts   : roc_curves.png, confusion_matrices.png, model/",
        ]

    lines += [
        "",
        "--- Best Run Summary ---",
        f"Model       : {best_name}",
        f"run_id      : {run_ids.get(run_key_map[best_name], 'unknown')}",
        f"CV ROC-AUC  : {cv_map[best_name]['roc_auc_mean']:.4f}",
        f"Test ROC-AUC: {results[best_name]['roc_auc']:.4f}",
        "",
        "Status : COMPLETE",
    ]
    write_report("step_3_mlflow", lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run():
    print("=== train.py ===")

    # MLflow setup (Phase 3) -- REPO_ROOT anchor avoids spaces/colon in abs path
    mlflow.set_tracking_uri(MLFLOW_DIR)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"MLflow tracking URI : {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment   : {MLFLOW_EXPERIMENT}")

    X_train, X_test, y_train, y_test = load_splits()

    # Tune all three models
    lr_pipe,  lr_params,  lr_cv_auc  = tune_logistic(X_train, y_train)
    rf_pipe,  rf_params,  rf_cv_auc  = tune_random_forest(X_train, y_train)
    xgb_pipe, xgb_params, xgb_cv_auc = tune_xgboost(X_train, y_train)

    # Full CV scores
    print("\n--- Full CV scores ---")
    cv_map = {
        "Logistic Regression": cv_scores(lr_pipe,  X_train, y_train, "Logistic Regression"),
        "Random Forest":       cv_scores(rf_pipe,  X_train, y_train, "Random Forest"),
        "XGBoost":             cv_scores(xgb_pipe, X_train, y_train, "XGBoost"),
    }

    # Test-set evaluation
    print("\n--- Test-set evaluation ---")
    lr_metrics,  lr_pred,  lr_proba  = evaluate_on_test(lr_pipe,  X_test, y_test, "Logistic Regression")
    rf_metrics,  rf_pred,  rf_proba  = evaluate_on_test(rf_pipe,  X_test, y_test, "Random Forest")
    xgb_metrics, xgb_pred, xgb_proba = evaluate_on_test(xgb_pipe, X_test, y_test, "XGBoost")

    results = {
        "Logistic Regression": lr_metrics,
        "Random Forest":       rf_metrics,
        "XGBoost":             xgb_metrics,
    }

    # Model selection: highest CV ROC-AUC
    cv_aucs = {
        "Logistic Regression": cv_map["Logistic Regression"]["roc_auc_mean"],
        "Random Forest":       cv_map["Random Forest"]["roc_auc_mean"],
        "XGBoost":             cv_map["XGBoost"]["roc_auc_mean"],
    }
    best_name = max(cv_aucs, key=cv_aucs.get)
    best_pipe = {"Logistic Regression": lr_pipe, "Random Forest": rf_pipe, "XGBoost": xgb_pipe}[best_name]
    best_params = {"Logistic Regression": lr_params, "Random Forest": rf_params, "XGBoost": xgb_params}[best_name]
    print(f"\nBest model (CV ROC-AUC): {best_name} ({cv_aucs[best_name]:.4f})")

    # Plots
    os.makedirs(SHOTS_DIR, exist_ok=True)
    models_data = [
        ("Logistic Regression", lr_pipe,  "#66c2a5"),
        ("Random Forest",       rf_pipe,  "#fc8d62"),
        ("XGBoost",             xgb_pipe, "#8da0cb"),
    ]
    plot_roc_curves(models_data, X_test, y_test)
    plot_confusion_matrices(models_data, X_test, y_test)

    # Save artifacts
    _, _, model_size, pipeline_size = save_artifacts(best_pipe)

    # MLflow: log all 3 runs (Phase 3)
    print("\n--- MLflow logging ---")
    artifact_paths = [
        os.path.join(SHOTS_DIR, "roc_curves.png"),
        os.path.join(SHOTS_DIR, "confusion_matrices.png"),
    ]
    run_key_map = {
        "Logistic Regression": "logistic_regression",
        "Random Forest":       "random_forest",
        "XGBoost":             "xgboost",
    }
    pipes   = {"Logistic Regression": lr_pipe,  "Random Forest": rf_pipe,  "XGBoost": xgb_pipe}
    params  = {"Logistic Regression": lr_params, "Random Forest": rf_params, "XGBoost": xgb_params}
    metrics = {"Logistic Regression": lr_metrics, "Random Forest": rf_metrics, "XGBoost": xgb_metrics}

    run_ids = {}
    for display_name, run_key in run_key_map.items():
        run_ids[run_key] = log_run(
            run_name=run_key,
            pipeline=pipes[display_name],
            params=params[display_name],
            cv=cv_map[display_name],
            test_metrics=metrics[display_name],
            artifact_paths=artifact_paths,
            X_train=X_train,
            is_best=(display_name == best_name),
        )

    # Output reports
    _write_report(results, best_name, best_params, cv_map, model_size, pipeline_size)
    _write_mlflow_report(run_ids, cv_map, results, best_name)
    print("train.py complete.")


if __name__ == "__main__":
    run()
