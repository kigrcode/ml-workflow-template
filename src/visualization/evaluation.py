"""
Evaluation visualisation utilities.
Used by the final evaluation notebook to generate
diagnostic plots for both classification and regression.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from pathlib import Path


# =========================================================
# CLASSIFICATION PLOTS
# =========================================================

def plot_roc_curve(y_test, y_proba, save_path: Path = None):
    """
    Plot ROC curve with AUC score.
    y_proba: predicted probabilities for positive class.
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return roc_auc


def plot_precision_recall_curve(y_test, y_proba, save_path: Path = None):
    """
    Plot Precision-Recall curve.
    y_proba: predicted probabilities for positive class.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, linewidth=2, color="purple")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_confusion_matrix(y_test, y_pred, save_path: Path = None):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_feature_importance(model, feature_names: list, save_path: Path = None):
    """
    Plot built-in feature importance for tree-based models.
    Falls back gracefully if model doesn't support feature_importances_.
    Note: SHAP importance is preferred — use this as a fallback only.
    """
    if not hasattr(model, "feature_importances_"):
        print("Model does not have feature_importances_. "
              "Use SHAP analysis for feature importance instead.")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)
    top_idx = idx[-20:]

    plt.figure(figsize=(8, 6))
    plt.barh(
        range(len(top_idx)),
        importances[top_idx],
        color="#378ADD"
    )
    plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
    plt.title("Feature Importance (top 20)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# =========================================================
# REGRESSION PLOTS
# =========================================================

def plot_predicted_vs_actual(y_test, y_pred, save_path: Path = None):
    """Scatter plot of predicted vs actual values."""
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, color="#378ADD")
    plt.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        "r--", linewidth=1
    )
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_residuals(y_test, y_pred, save_path: Path = None):
    """Residuals vs predicted values."""
    residuals = np.array(y_test) - np.array(y_pred)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.6, color="#378ADD")
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_residual_distribution(y_test, y_pred, save_path: Path = None):
    """Distribution of residuals."""
    residuals = np.array(y_test) - np.array(y_pred)

    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True, color="#378ADD")
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def regression_summary(y_test, y_pred) -> dict:
    """
    Compute and print regression metrics summary.
    Returns metrics dict for use in reports.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return metrics
