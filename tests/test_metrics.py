"""
Tests for src/models/training/metrics.py

Verifies that classification and regression metrics compute
correctly and handle edge cases properly.
"""

import numpy as np
import pytest

from src.models.training.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    get_primary_metric_name,
    extract_primary_metric,
)


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def binary_classification_data():
    """Perfect predictions for easy verification."""
    y_true  = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred  = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.9, 0.1])
    return y_true, y_pred, y_proba


@pytest.fixture
def imperfect_classification_data():
    """Realistic imperfect predictions."""
    y_true  = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred  = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.6, 0.8, 0.4, 0.3, 0.9, 0.2])
    return y_true, y_pred, y_proba


@pytest.fixture
def regression_data():
    """Perfect regression predictions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    return y_true, y_pred


# =========================================================
# TESTS — compute_classification_metrics
# =========================================================

def test_classification_metrics_returns_expected_keys(binary_classification_data):
    y_true, y_pred, y_proba = binary_classification_data
    metrics = compute_classification_metrics(y_true, y_proba, y_pred)
    expected_keys = {"roc_auc", "f1", "precision", "recall", "accuracy", "log_loss"}
    assert expected_keys.issubset(set(metrics.keys()))


def test_classification_metrics_perfect_predictions(binary_classification_data):
    y_true, y_pred, y_proba = binary_classification_data
    metrics = compute_classification_metrics(y_true, y_proba, y_pred)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_classification_metrics_values_in_range(imperfect_classification_data):
    y_true, y_pred, y_proba = imperfect_classification_data
    metrics = compute_classification_metrics(y_true, y_proba, y_pred)
    for key in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        assert 0.0 <= metrics[key] <= 1.0, \
            f"{key} value {metrics[key]} is out of range"


def test_classification_metrics_no_exceptions_on_edge_cases():
    y_true  = np.array([0, 1, 0, 1])
    y_pred  = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = compute_classification_metrics(y_true, y_proba, y_pred)
    assert isinstance(metrics, dict)


# =========================================================
# TESTS — compute_regression_metrics
# =========================================================

def test_regression_metrics_returns_expected_keys(regression_data):
    y_true, y_pred = regression_data
    metrics = compute_regression_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"rmse", "mae", "r2"}


def test_regression_metrics_perfect_predictions():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_regression_metrics_values_are_non_negative(regression_data):
    y_true, y_pred = regression_data
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0


# =========================================================
# TESTS — get_primary_metric_name
# =========================================================

def test_primary_metric_classification():
    assert get_primary_metric_name("classification") == "roc_auc"


def test_primary_metric_regression():
    assert get_primary_metric_name("regression") == "rmse"


def test_primary_metric_unknown_task_raises():
    with pytest.raises(ValueError):
        get_primary_metric_name("unknown_task")


# =========================================================
# TESTS — extract_primary_metric
# =========================================================

def test_extract_primary_metric_classification():
    metrics = {"roc_auc": 0.85, "f1": 0.80}
    name, value = extract_primary_metric("classification", metrics)
    assert name == "roc_auc"
    assert value == pytest.approx(0.85)


def test_extract_primary_metric_regression():
    metrics = {"rmse": 0.25, "mae": 0.18, "r2": 0.92}
    name, value = extract_primary_metric("regression", metrics)
    assert name == "rmse"
    assert value == pytest.approx(0.25)


def test_extract_primary_metric_missing_key_returns_nan():
    metrics = {"f1": 0.80}
    name, value = extract_primary_metric("classification", metrics)
    assert name == "roc_auc"
    assert np.isnan(value)