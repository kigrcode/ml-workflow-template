import pandas as pd
from pathlib import Path
from src.config.paths import RAW_DATA_DIR


def load_raw_data(path: str | Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Load raw data from a file or directory.

    If a directory is provided, automatically load the first CSV file inside it.
    """
    path = Path(path)

    # If a directory is passed, auto-detect the first CSV file
    if path.is_dir():
        csv_files = list(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {path}")
        if len(csv_files) > 1:
            print(f"Multiple CSV files found in {path}, loading the first one: {csv_files[0].name}")
        path = csv_files[0]

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)



