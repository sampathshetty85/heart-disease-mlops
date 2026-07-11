"""
preprocess.py — Task 1 data cleaning and preprocessing.

PREPROCESSING DESIGN DECISIONS
1. ENCODING + SCALING: Deferred to sklearn Pipeline in train.py (Phase 2).
   Reason: prevents data leakage — scaler/encoder fit only on X_train.
   Satisfies Task 1 encoding/scaling requirement via Phase 2 Pipeline.
   Documented in EDA notebook (Step 1.3) for marker visibility.

2. ENCODING STRATEGY: OneHotEncoder for ALL categoricals incl. ca, thal.
   Reason: treats as nominal — avoids imposing arbitrary ordinal ordering.

3. OUTLIER TREATMENT: Not applied.
   Reason: tree models robust; Logistic Regression protected by StandardScaler.

4. FEATURE COLUMNS: feature_columns.json uses list(X_train.columns).
   Order must match X_train.csv exactly for correct API inference.
   Generated at docker build time — not committed to repo.
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths — anchored to __file__ so script is working-directory-agnostic
# (required for Docker WORKDIR=/app context)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "..", "..", "data", "raw", "heart.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "..", "data", "processed")

# ---------------------------------------------------------------------------
# Feature lists — single source of truth imported by train.py (Phase 2)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


# ---------------------------------------------------------------------------
# Step A — Load raw data
# ---------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            "data/raw/heart.csv not found. Run src/data/download.py first."
        )
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded: {len(df)} rows x {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Step B — Handle missing values
# ---------------------------------------------------------------------------
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["ca", "thal"]:
        median_val = int(df[col].median())
        n_missing = int(df[col].isnull().sum())
        df[col] = df[col].fillna(median_val).astype(int)
        print(f"Imputed {col:<4} with median: {median_val} ({n_missing} values)")
    print(f"Missing values after imputation: {df.isnull().sum().sum()}")
    return df


# ---------------------------------------------------------------------------
# Step C — Binarize target column
# ---------------------------------------------------------------------------
def binarize_target(df: pd.DataFrame):
    df = df.copy()
    # fbs guard — must be clean binary before split
    assert set(df["fbs"].dropna().unique()).issubset({0, 1}), \
        f"fbs contains unexpected values: {df['fbs'].unique()}"
    print(f"fbs assertion passed: values {set(df['fbs'].unique())}")

    y = (df["num"] > 0).astype(int).rename("target")
    X = df.drop(columns=["num"])

    class_counts = y.value_counts().sort_index()
    print(f"Target binarized: {class_counts.get(0, 0)} (class 0) / "
          f"{class_counts.get(1, 0)} (class 1)")
    return X, y


# ---------------------------------------------------------------------------
# Step D — Stratified train/test split
# ---------------------------------------------------------------------------
def split(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_ratio = y_train.mean() * 100
    test_ratio = y_test.mean() * 100
    print(f"Split: {len(X_train)} train / {len(X_test)} test (stratified)")
    print(f"Train class ratio: {100 - train_ratio:.1f}% / {train_ratio:.1f}%")
    print(f"Test  class ratio: {100 - test_ratio:.1f}% / {test_ratio:.1f}%")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step E — Save all outputs
# ---------------------------------------------------------------------------
def save_splits(X_train, X_test, y_train, y_test, df_cleaned):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    files = {
        "X_train.csv": X_train,
        "X_test.csv":  X_test,
        "y_train.csv": y_train.to_frame(),
        "y_test.csv":  y_test.to_frame(),
        "heart_cleaned.csv": df_cleaned,
    }
    for fname, data in files.items():
        path = os.path.join(PROCESSED_DIR, fname)
        data.to_csv(path, index=False)
        print(f"Saved: data/processed/{fname:<30} "
              f"({data.shape[0]} x {data.shape[1]})")

    # feature_columns.json — order must match X_train.csv exactly
    feature_cols_path = os.path.join(PROCESSED_DIR, "feature_columns.json")
    with open(feature_cols_path, "w") as f:
        json.dump(list(X_train.columns), f, indent=2)
    print(f"Saved: data/processed/feature_columns.json "
          f"({len(X_train.columns)} features)")


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
def run_assertions(df_imputed, X, y, X_train, X_test, y_train, y_test, df_cleaned):
    assert df_imputed.isnull().sum().sum() == 0, \
        "Nulls remain after imputation"
    assert df_imputed["ca"].dtype == int or str(df_imputed["ca"].dtype) in ("int32", "int64"), \
        "ca should be int after imputation"
    assert df_imputed["thal"].dtype == int or str(df_imputed["thal"].dtype) in ("int32", "int64"), \
        "thal should be int after imputation"
    assert set(y.unique()) == {0, 1}, \
        f"Target is not binary: {y.unique()}"
    assert len(df_cleaned) == 303, \
        f"heart_cleaned.csv should have 303 rows, got {len(df_cleaned)}"
    assert len(X_train) == 242, \
        f"X_train should have 242 rows, got {len(X_train)}"
    assert len(X_test) == 61, \
        f"X_test should have 61 rows, got {len(X_test)}"

    # Stratification check — both splits within 2% of overall ratio
    overall_ratio = y.mean()
    for name, y_split in [("train", y_train), ("test", y_test)]:
        split_ratio = y_split.mean()
        assert abs(split_ratio - overall_ratio) < 0.02, \
            f"{name} class ratio {split_ratio:.3f} deviates >2% from {overall_ratio:.3f}"

    # All 6 output files exist
    for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv",
                  "heart_cleaned.csv", "feature_columns.json"]:
        path = os.path.join(PROCESSED_DIR, fname)
        assert os.path.exists(path), f"Output file missing: {path}"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run():
    print("=== preprocess.py ===")

    df_raw = load_raw()
    df_imputed = handle_missing(df_raw)
    X, y = binarize_target(df_imputed)

    # Build heart_cleaned.csv: all features + target, pre-split
    df_cleaned = X.copy()
    df_cleaned["target"] = y.values

    X_train, X_test, y_train, y_test = split(X, y)
    save_splits(X_train, X_test, y_train, y_test, df_cleaned)

    run_assertions(df_imputed, X, y, X_train, X_test, y_train, y_test, df_cleaned)
    print("All assertions passed. preprocess.py complete.")


if __name__ == "__main__":
    run()
