"""
Tests for src/features/type_detection.py

Verifies that detect_feature_types correctly classifies
numeric, binary, categorical, high cardinality and datetime columns
using synthetic data — no dependency on any real dataset.
"""

import pandas as pd
import numpy as np
import pytest

from src.features.type_detection import (
    detect_feature_types,
    is_numeric,
    is_binary,
    is_categorical,
    is_datetime,
    is_constant,
    is_near_constant,
    missing_ratio,
)


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def sample_df():
    """
    Synthetic dataframe covering all feature types.
    Used across multiple tests.
    """
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        "numeric_col":       np.random.randn(n),
        "binary_col":        np.random.choice([0, 1], n),
        "categorical_col":   np.random.choice(["a", "b", "c"], n),
        "high_card_col":     [f"id_{i}" for i in range(n)],
        "datetime_col":      pd.date_range("2020-01-01", periods=n, freq="D"),
        "constant_col":      [1] * n,
        "near_constant_col": [1] * 99 + [2],
    })


# =========================================================
# TESTS — detect_feature_types
# =========================================================

def test_numeric_detected(sample_df):
    result = detect_feature_types(sample_df)
    assert "numeric_col" in result["numeric"]


def test_binary_detected(sample_df):
    result = detect_feature_types(sample_df)
    assert "binary_col" in result["binary"]


def test_categorical_detected(sample_df):
    result = detect_feature_types(sample_df)
    assert "categorical_col" in result["categorical"]


def test_high_cardinality_detected(sample_df):
    result = detect_feature_types(sample_df)
    assert "high_card_col" in result["high_cardinality"]


def test_datetime_detected(sample_df):
    result = detect_feature_types(sample_df)
    assert "datetime_col" in result["datetime"]


def test_all_keys_present(sample_df):
    result = detect_feature_types(sample_df)
    assert set(result.keys()) == {
        "numeric", "binary", "categorical", "high_cardinality", "datetime"
    }


def test_no_column_appears_twice(sample_df):
    result = detect_feature_types(sample_df)
    all_cols = (
        result["numeric"] +
        result["binary"] +
        result["categorical"] +
        result["high_cardinality"] +
        result["datetime"]
    )
    assert len(all_cols) == len(set(all_cols))


# =========================================================
# TESTS — helper functions
# =========================================================

def test_is_numeric():
    assert is_numeric(pd.Series([1.0, 2.0, 3.0]))
    assert not is_numeric(pd.Series(["a", "b", "c"]))


def test_is_binary():
    assert is_binary(pd.Series([0, 1, 0, 1]))
    assert not is_binary(pd.Series([0, 1, 2, 3]))


def test_is_categorical():
    assert is_categorical(pd.Series(["a", "b", "c"]))
    assert is_categorical(pd.Series(pd.Categorical(["a", "b", "c"])))
    assert not is_categorical(pd.Series([1.0, 2.0, 3.0]))


def test_is_datetime():
    assert is_datetime(pd.Series(pd.date_range("2020-01-01", periods=5)))
    assert not is_datetime(pd.Series([1.0, 2.0, 3.0]))
    assert not is_datetime(pd.Series(["not", "a", "date", "at", "all"]))


def test_is_near_constant():
    assert is_near_constant(pd.Series([1] * 99 + [2]), threshold=0.98)
    assert not is_near_constant(pd.Series([1, 2, 3, 4] * 25), threshold=0.98)


def test_is_near_constant():
    assert bool(is_near_constant(pd.Series([1] * 99 + [2]), threshold=0.98))
    assert not bool(is_near_constant(pd.Series([1, 2, 3, 4] * 25), threshold=0.98))


def test_missing_ratio():
    s = pd.Series([1, None, None, None, 1]) 
    assert missing_ratio(s) == pytest.approx(0.6)


def test_missing_ratio_no_missing():
    s = pd.Series([1, 2, 3, 4, 5])
    assert missing_ratio(s) == 0.0