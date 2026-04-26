"""
Reusable SHAP analysis utilities.
"""

import shap
import pandas as pd
from rich import print


def compute_shap_values(model, X: pd.DataFrame):
    """
    Compute SHAP values for any tree-based or linear model.
    Falls back to KernelExplainer for unsupported models.
    """
    print("[bold green]Computing SHAP values...[/bold green]")

    try:
        explainer = shap.Explainer(model, X)
    except Exception:
        explainer = shap.KernelExplainer(model.predict, X)

    shap_values = explainer(X)
    return explainer, shap_values


def plot_summary(shap_values, X: pd.DataFrame, save_path: str | None = None):
    """
    Generate a SHAP summary plot.
    Optionally save it to disk.
    """
    print("[bold blue]Generating SHAP summary plot...[/bold blue]")

    shap.summary_plot(shap_values.values, X, show=False)

    if save_path:
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[bold green]Saved SHAP summary to:[/bold green] {save_path}")       