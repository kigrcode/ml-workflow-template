import pandas as pd
from pathlib import Path
from src.config.paths import RAW_DATA_DIR


# String-based missing value indicators that are safe to handle
# automatically across all datasets
DEFAULT_NA_VALUES = ["?", "name", "n/a", "N/A", "na", "NA", "none", "None", "missing"]


def load_raw_data(
    path: str | Path = RAW_DATA_DIR,
    na_values: list = None
) -> pd.DataFrame:
    """
    Load raw data from a file or directory.

    Automatically handles common string-based missing value indicators.
    Numeric sentinels (e.g. -9, -999) are NOT handled automatically
    as they may be valid values in some datasets — specify them
    explicitly via na_values in config.yaml instead.

    If a directory is provided, automatically loads the first CSV file.
    na_values: additional dataset-specific missing value indicators
               loaded from config.yaml
    """
    path = Path(path)

    if path.is_dir():
        csv_files = list(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {path}")
        if len(csv_files) > 1:
            print(f"Multiple CSV files found, loading first: {csv_files[0].name}")
        path = csv_files[0]

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    all_na_values = DEFAULT_NA_VALUES + (na_values or [])

    return pd.read_csv(path, na_values=all_na_values, keep_default_na=True)


def prepare_data(df: pd.DataFrame, config: dict):
    """
    Apply dataset-specific preparation steps from config:
    - Binarise target if configured
    - Convert specified columns to categorical

    Returns X, y ready for splitting.
    """
    TARGET = config["target"]
    DROP_COLS = config.get("drop_columns", [])

    X = df.drop(columns=[TARGET] + DROP_COLS)
    y = df[TARGET]

    # Binarise target if configured
    if config.get("binarise_target", False):
        y = (y > 0).astype(int)
        print(f"Target binarised — class distribution:\n{y.value_counts()}")

    # Convert specified columns to categorical
    categorical_cols = config.get("categorical_columns", [])
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)

    if categorical_cols:
        print(f"Converted to categorical: {categorical_cols}")

    return X, y



