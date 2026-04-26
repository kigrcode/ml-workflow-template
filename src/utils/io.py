"""
General-purpose IO utilities for safe, explicit file handling.
"""

from pathlib import Path
import pandas as pd
from rich import print


def ensure_dir(path: str | Path) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file with logging."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    print(f"[bold green]Loading:[/bold green] {path}")
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """Save a DataFrame to CSV with directory safety."""
    path = Path(path)
    ensure_dir(path.parent)
    print(f"[bold blue]Saving:[/bold blue] {path}")
    df.to_csv(path, index=index)