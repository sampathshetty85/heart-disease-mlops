"""
test_preprocess.py -- unit tests for src/data/preprocess.py
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.preprocess import (  # noqa: E402
    load_raw, handle_missing, binarize_target, split,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW_READY = os.path.exists(os.path.join(REPO_ROOT, "data", "raw", "heart.csv"))


@pytest.mark.skipif(not RAW_READY, reason="Run src/data/download.py first")
def test_load_raw_returns_dataframe():
    df = load_raw()
    assert df.shape == (303, 14), f"Expected (303, 14), got {df.shape}"


@pytest.mark.skipif(not RAW_READY, reason="Run src/data/download.py first")
def test_handle_missing_no_nulls():
    df = load_raw()
    df_clean = handle_missing(df)
    assert df_clean.isnull().sum().sum() == 0


@pytest.mark.skipif(not RAW_READY, reason="Run src/data/download.py first")
def test_handle_missing_ca_thal_dtype():
    df = load_raw()
    df_clean = handle_missing(df)
    assert str(df_clean["ca"].dtype) in ("int32", "int64")
    assert str(df_clean["thal"].dtype) in ("int32", "int64")


@pytest.mark.skipif(not RAW_READY, reason="Run src/data/download.py first")
def test_binarize_target_binary():
    df = load_raw()
    df_clean = handle_missing(df)
    _, y = binarize_target(df_clean)
    assert set(y.unique()) == {0, 1}


@pytest.mark.skipif(not RAW_READY, reason="Run src/data/download.py first")
def test_binarize_target_drops_num():
    df = load_raw()
    df_clean = handle_missing(df)
    X, _ = binarize_target(df_clean)
    assert "num" not in X.columns


@pytest.mark.skipif(not RAW_READY, reason="Run src/data/download.py first")
def test_split_shapes():
    df = load_raw()
    df_clean = handle_missing(df)
    X, y = binarize_target(df_clean)
    X_train, X_test, y_train, y_test = split(X, y)
    assert len(X_train) == 242
    assert len(X_test) == 61
    assert len(y_train) == 242
    assert len(y_test) == 61


@pytest.mark.skipif(not RAW_READY, reason="Run src/data/download.py first")
def test_split_stratification():
    df = load_raw()
    df_clean = handle_missing(df)
    X, y = binarize_target(df_clean)
    X_train, X_test, y_train, y_test = split(X, y)
    overall = y.mean()
    assert abs(y_train.mean() - overall) < 0.02
    assert abs(y_test.mean() - overall) < 0.02


def test_feature_columns_exported():
    assert len(NUMERIC_FEATURES) == 5
    assert len(CATEGORICAL_FEATURES) == 8
    assert len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) == 13
