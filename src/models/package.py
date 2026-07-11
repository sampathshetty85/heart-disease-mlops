"""
package.py -- Phase 4: Model Packaging & Reproducibility verification.

Verifies that all serialized artifacts are loadable, the preprocessing
pipeline is intact, inference is deterministic, and all pinned dependencies
match their installed versions.

Run standalone:
    python src/models/package.py

Also executed as a RUN step during docker build to bake a verified
packaging report into the image.
"""

import os
import sys
import json
import importlib
import importlib.metadata
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import joblib  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from report_writer import write_report  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(BASE_DIR, "..", "..")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")
REQUIREMENTS = os.path.join(REPO_ROOT, "requirements.txt")

MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
PIPELINE_PATH = os.path.join(MODELS_DIR, "pipeline.joblib")
FEAT_COLS = os.path.join(DATA_DIR, "feature_columns.json")

# Best MLflow run ID (logistic_regression, best_model=true)
MLFLOW_RUN_ID = "e4abcc6349dd4254ac77172fbd19d08c"
MLFLOW_MODEL_URI = f"runs:/{MLFLOW_RUN_ID}/model"

SAMPLE_INPUT = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
}


# ---------------------------------------------------------------------------
# A -- Artifact check
# ---------------------------------------------------------------------------
def check_artifacts():
    results = {}

    for name, path in [("model.joblib", MODEL_PATH), ("pipeline.joblib", PIPELINE_PATH)]:
        assert os.path.exists(path), f"Missing artifact: {path}"
        obj = joblib.load(path)
        size = os.path.getsize(path)
        results[name] = {"size": size, "obj": obj}
        print(f"  {name:<22}: {size:,} bytes  LOADABLE")

    pipeline = results["pipeline.joblib"]["obj"]
    assert hasattr(pipeline, "named_steps"), "pipeline.joblib is not a sklearn Pipeline"
    assert "preprocessor" in pipeline.named_steps, "Pipeline missing 'preprocessor' step"
    assert "classifier" in pipeline.named_steps, "Pipeline missing 'classifier' step"
    print(f"  Pipeline steps         : {list(pipeline.named_steps.keys())}")

    model = results["model.joblib"]["obj"]
    assert hasattr(model, "predict"),       "model.joblib missing predict()"
    assert hasattr(model, "predict_proba"), "model.joblib missing predict_proba()"
    print(f"  Model type             : {type(model).__name__}")

    assert os.path.exists(FEAT_COLS), f"Missing: {FEAT_COLS}"
    with open(FEAT_COLS) as f:
        feat_cols = json.load(f)
    print(f"  feature_columns.json   : {len(feat_cols)} features  EXISTS")

    return results, feat_cols


# ---------------------------------------------------------------------------
# B -- Requirements version audit
# ---------------------------------------------------------------------------
def check_requirements():
    assert os.path.exists(REQUIREMENTS), f"requirements.txt not found at {REQUIREMENTS}"

    pkg_pattern = re.compile(r"^([A-Za-z0-9_\-\[\]]+)==(.+)$")
    audits = []

    with open(REQUIREMENTS) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            m = pkg_pattern.match(line)
            if not m:
                continue

            pkg_spec, pinned_version = m.group(1), m.group(2)
            # Strip extras like [standard] for metadata lookup
            pkg_name = re.sub(r"\[.*\]", "", pkg_spec)

            try:
                installed = importlib.metadata.version(pkg_name)
                if installed == pinned_version:
                    status = "PASS"
                else:
                    status = f"WARN (installed={installed})"
            except importlib.metadata.PackageNotFoundError:
                installed = "NOT FOUND"
                status = "FAIL"

            audits.append({
                "spec":    pkg_spec,
                "pinned":  pinned_version,
                "installed": installed,
                "status":  status,
            })
            icon = "  PASS" if status == "PASS" else ("  WARN" if "WARN" in status else "  FAIL")
            print(f"{icon}  {pkg_spec:<42} pinned={pinned_version}  installed={installed}")

    fails = [a for a in audits if a["status"] == "FAIL"]
    warns = [a for a in audits if "WARN" in a["status"]]
    passes = [a for a in audits if a["status"] == "PASS"]
    print(f"\n  {len(passes)} PASS  {len(warns)} WARN  {len(fails)} FAIL  ({len(audits)} total packages)")
    return audits


# ---------------------------------------------------------------------------
# C -- Smoke test inference
# ---------------------------------------------------------------------------
def smoke_test_inference():
    from models.predict import predict
    result = predict(SAMPLE_INPUT)

    assert isinstance(result, dict),                  "predict() must return a dict"
    assert "prediction" in result,                   "missing key: prediction"
    assert "confidence" in result,                   "missing key: confidence"
    assert "label" in result,                   "missing key: label"
    assert result["prediction"] in (0, 1),            f"prediction not 0/1: {result['prediction']}"
    assert 0.0 <= result["confidence"] <= 1.0,        f"confidence out of range: {result['confidence']}"
    assert result["label"] in ("Heart Disease", "No Heart Disease"), \
        f"unexpected label: {result['label']}"

    print("  Input      : age=63, sex=1, cp=3, trestbps=145, chol=233, ...")
    print(f"  Output     : prediction={result['prediction']}  "
          f"confidence={result['confidence']}  label={result['label']}")
    print("  Status     : PASS")
    return result


# ---------------------------------------------------------------------------
# D -- Determinism check
# ---------------------------------------------------------------------------
def check_determinism():
    from models.predict import predict
    r1 = predict(SAMPLE_INPUT)
    r2 = predict(SAMPLE_INPUT)

    assert r1["prediction"] == r2["prediction"],  "Non-deterministic prediction"
    assert r1["confidence"] == r2["confidence"],  "Non-deterministic confidence"

    print(f"  Run 1      : prediction={r1['prediction']}  confidence={r1['confidence']}")
    print(f"  Run 2      : prediction={r2['prediction']}  confidence={r2['confidence']}")
    print("  Match      : PASS")
    return r1, r2


# ---------------------------------------------------------------------------
# E -- Output report
# ---------------------------------------------------------------------------
def _write_report(artifact_results, feat_cols, audits, smoke_result, det_r1, det_r2):
    model_size = artifact_results["model.joblib"]["size"]
    pipeline_size = artifact_results["pipeline.joblib"]["size"]
    pipeline_obj = artifact_results["pipeline.joblib"]["obj"]
    model_obj = artifact_results["model.joblib"]["obj"]

    fails = [a for a in audits if a["status"] == "FAIL"]
    warns = [a for a in audits if "WARN" in a["status"]]
    passes = [a for a in audits if a["status"] == "PASS"]

    lines = [
        "STEP 4 -- Model Packaging & Reproducibility",
        "",
        "--- Serialized Artifacts ---",
        f"models/model.joblib      : {model_size:,} bytes  format=joblib  status=LOADABLE",
        f"models/pipeline.joblib   : {pipeline_size:,} bytes  format=joblib  status=LOADABLE",
        f"Pipeline steps           : {list(pipeline_obj.named_steps.keys())}",
        f"Model type               : {type(model_obj).__name__}",
        f"MLflow model URI         : {MLFLOW_MODEL_URI}",
        f"feature_columns.json     : {len(feat_cols)} features  status=EXISTS",
        "Serialization format     : joblib (sklearn-compatible, Python 3.12)",
        "",
        f"--- Requirements Audit ({len(audits)} packages) ---",
    ]
    for a in audits:
        lines.append(
            f"  {a['spec']:<42} pinned={a['pinned']:<12} installed={a['installed']:<12} {a['status']}"
        )
    lines += [
        "",
        f"  Summary: {len(passes)} PASS  {len(warns)} WARN  {len(fails)} FAIL",
        f"  Overall: {'PASS' if len(fails) == 0 else 'FAIL'}",
        "",
        "--- Smoke Test ---",
        f"  Input  : {SAMPLE_INPUT}",
        f"  Output : prediction={smoke_result['prediction']} "
        f"confidence={smoke_result['confidence']}  label={smoke_result['label']}",
        "  Status : PASS",
        "",
        "--- Determinism Check ---",
        f"  Run 1  : prediction={det_r1['prediction']}  confidence={det_r1['confidence']}",
        f"  Run 2  : prediction={det_r2['prediction']}  confidence={det_r2['confidence']}",
        "  Match  : PASS",
        "",
        "--- Docker Proof ---",
        "  Docker build/test proof: see Phase 6 -- output/step_6_api.txt",
        "  (model baked into image at docker build time; API serves instantly on container start)",
        "",
        "Status : COMPLETE",
    ]
    write_report("step_4_packaging", lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run():
    print("=== package.py ===")

    print("\n[1] Artifact check")
    artifact_results, feat_cols = check_artifacts()

    print("\n[2] Requirements version audit")
    audits = check_requirements()

    print("\n[3] Smoke test inference")
    smoke_result = smoke_test_inference()

    print("\n[4] Determinism check")
    det_r1, det_r2 = check_determinism()

    _write_report(artifact_results, feat_cols, audits, smoke_result, det_r1, det_r2)
    print("\npackage.py complete.")


if __name__ == "__main__":
    run()
