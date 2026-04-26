"""
Threshold tuning for probabilistic classifiers.
Only applies to classification problems — skipped automatically for regression.
Optimal threshold is saved to artifacts for use in final evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


METRIC_FUNCTIONS = {
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
    "accuracy": accuracy_score,
}


def tune_threshold(
    y_true,
    y_proba,
    metric_name: str,
    num_thresholds: int,
) -> dict:
    """
    Search over thresholds in [0, 1] to maximise a given metric.
    Assumes binary classification and y_proba is probability for class 1.

    metric_name: one of f1, precision, recall, accuracy
    num_thresholds: number of thresholds to evaluate
    """
    if metric_name not in METRIC_FUNCTIONS:
        raise ValueError(
            f"Unknown metric: {metric_name}. "
            f"Options are: {list(METRIC_FUNCTIONS.keys())}"
        )

    metric_fn = METRIC_FUNCTIONS[metric_name]
    thresholds = np.linspace(0, 1, num_thresholds)

    best_score = -np.inf
    best_threshold = 0.5

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        try:
            score = metric_fn(y_true, y_pred, zero_division=0) \
                if metric_name in ["precision", "recall"] \
                else metric_fn(y_true, y_pred)
            if score > best_score:
                best_score = score
                best_threshold = t
        except Exception:
            continue

    return {
        "best_threshold": float(best_threshold),
        "best_score": float(best_score),
        "metric": metric_name,
    }


def compute_threshold_curves(y_true, y_proba, num_thresholds: int = 101) -> pd.DataFrame:
    """
    Compute precision, recall, f1 and accuracy across all thresholds.
    Used for plotting the precision-recall tradeoff curve.
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    rows = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 4),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
        })

    return pd.DataFrame(rows)