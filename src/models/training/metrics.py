from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    precision_score,
    recall_score,
)


CLASSIFICATION_PRIMARY = "roc_auc"
REGRESSION_PRIMARY = "rmse"


def compute_classification_metrics(y_true, y_proba, y_pred) -> Dict[str, float]:
    """
    y_proba: predicted probabilities for positive class (binary) or full matrix (multiclass)
    y_pred: predicted class labels
    """
    metrics: Dict[str, float] = {}

    # ROC-AUC
    try:
        if y_proba.ndim == 1:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        else:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
    except Exception:
        metrics["roc_auc"] = np.nan

    # F1
    try:
        metrics["f1"] = f1_score(y_true, y_pred, average="macro")
    except Exception:
        metrics["f1"] = np.nan

    # Precision
    try:
        metrics["precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    except Exception:
        metrics["precision"] = np.nan

    # Recall
    try:
        metrics["recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    except Exception:
        metrics["recall"] = np.nan

    # Accuracy
    try:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
    except Exception:
        metrics["accuracy"] = np.nan

    # Log loss
    try:
        metrics["log_loss"] = log_loss(y_true, y_proba)
    except Exception:
        metrics["log_loss"] = np.nan

    return metrics


def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    # RMSE — compatible with all sklearn versions
    try:
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    except Exception:
        metrics["rmse"] = np.nan

    try:
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
    except Exception:
        metrics["mae"] = np.nan

    try:
        metrics["r2"] = r2_score(y_true, y_pred)
    except Exception:
        metrics["r2"] = np.nan

    return metrics


def get_primary_metric_name(task: str) -> str:
    if task == "classification":
        return CLASSIFICATION_PRIMARY
    elif task == "regression":
        return REGRESSION_PRIMARY
    else:
        raise ValueError(f"Unknown task type: {task}")


def extract_primary_metric(task: str, metrics: Dict[str, float]) -> Tuple[str, float]:
    name = get_primary_metric_name(task)
    return name, metrics.get(name, np.nan)