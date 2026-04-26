"""
Model saving and loading utilities.
"""

from pathlib import Path
import joblib
from rich import print


def save_model(model, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[bold blue]Saving model:[/bold blue] {path}")
    joblib.dump(model, path)


def load_model(path: str | Path):
    return joblib.load(Path(path))