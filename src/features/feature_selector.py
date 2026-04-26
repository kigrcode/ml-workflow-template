"""
Two-stage feature selection:
1. Mutual information pre-filter — fast, removes obvious noise
2. SHAP-based refinement — precise, model-aware
"""

import numpy as np
import pandas as pd
import shap

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


# =========================================================
# STAGE 1 — MUTUAL INFORMATION PRE-FILTER
# =========================================================
def mutual_info_filter(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    percentile: int = 50,
) -> list:
    """
    Keep top features by mutual information score.
    percentile=50 keeps the top 50% of features.
    Returns list of selected feature names.
    """
    if task == "classification":
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42)

    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    threshold_idx = int(len(mi_series) * (percentile / 100))
    selected = mi_series.iloc[:threshold_idx].index.tolist()

    print(f"MI filter: {len(X.columns)} → {len(selected)} features "
          f"(top {percentile}%)")

    return selected, mi_series


# =========================================================
# STAGE 2 — SHAP REFINEMENT
# =========================================================
def shap_filter(
    model,
    X: pd.DataFrame,
    percentile: int = 80,
) -> list:
    """
    Keep top features by mean absolute SHAP value.
    percentile=80 keeps the top 80% of features.
    Returns list of selected feature names.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle multiclass output
    if isinstance(shap_values, list):
        shap_values = np.abs(np.array(shap_values)).mean(axis=0)

    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns
    ).sort_values(ascending=False)

    threshold_idx = int(len(mean_shap) * (percentile / 100))
    selected = mean_shap.iloc[:threshold_idx].index.tolist()

    print(f"SHAP filter: {len(X.columns)} → {len(selected)} features "
          f"(top {percentile}%)")

    return selected, mean_shap


# =========================================================
# ORCHESTRATOR
# =========================================================
def run_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model,
    task: str,
    config: dict,
) -> tuple:
    """
    Run two-stage feature selection.
    Stage 1 — MI filter on full feature set
    Stage 2 — SHAP filter on full feature set using same fitted model
    Final selection is the intersection of both
    """
    fs_config = config.get("feature_selection", {})
    mi_percentile = fs_config.get("mi_percentile", 50)
    shap_percentile = fs_config.get("shap_percentile", 80)

    # Stage 1 — MI filter on full feature set
    mi_selected, mi_scores = mutual_info_filter(
        X_train, y_train,
        task=task,
        percentile=mi_percentile
    )

    # Stage 2 — SHAP filter on full feature set
    # Model must have been trained on X_train so shapes match
    shap_selected, shap_scores = shap_filter(
        model,
        X_train,
        percentile=shap_percentile
    )

    # Final selection — intersection of MI and SHAP selected features
    final_selected = [f for f in shap_selected if f in mi_selected]

    print(f"Final selection (intersection): {len(final_selected)} features")

    return final_selected, {
        "mi_scores": mi_scores,
        "shap_scores": shap_scores,
        "mi_selected": mi_selected,
        "shap_selected": shap_selected,
    }