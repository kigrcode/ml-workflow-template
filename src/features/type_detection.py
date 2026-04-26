import pandas as pd
import numpy as np


# -----------------------------
# Basic Type Checks
# -----------------------------

def is_numeric(series: pd.Series) -> bool:
    """Return True if the series is numeric."""
    return pd.api.types.is_numeric_dtype(series)


def is_categorical(series: pd.Series) -> bool:
    """Return True if the series is categorical or object dtype."""
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or hasattr(series, 'cat')
        or pd.api.types.is_bool_dtype(series)
    )


def is_binary(series: pd.Series) -> bool:
    """Return True only for numeric binary columns."""
    return (
        pd.api.types.is_numeric_dtype(series) and
        series.dropna().nunique() == 2
    )



def is_datetime(series: pd.Series) -> bool:
    # Already datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # Only attempt string parsing on object dtype
    if not pd.api.types.is_object_dtype(series):
        return False

    # Try parsing strings as datetime
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    parsed = pd.to_datetime(non_null, errors="coerce")
    parse_rate = parsed.notna().mean()

    return parse_rate >= 0.8
 


# -----------------------------
# Cardinality Checks
# -----------------------------

def is_high_cardinality(series: pd.Series, threshold: int = 50) -> bool:
    """Return True if the number of unique values exceeds the threshold."""
    return series.nunique(dropna=True) > threshold


def is_low_cardinality(series: pd.Series, threshold: int = 20) -> bool:
    """Return True if the number of unique values is below or equal to the threshold."""
    return series.nunique(dropna=True) <= threshold


# -----------------------------
# Missingness Checks
# -----------------------------

def missing_ratio(series: pd.Series) -> float:
    """Return the proportion of missing values in the series."""
    return series.isna().mean()


def is_missing_heavy(series: pd.Series, threshold: float = 0.4) -> bool:
    """Return True if missingness exceeds the threshold."""
    return missing_ratio(series) > threshold


# -----------------------------
# Constant / Near-Constant
# -----------------------------

def is_constant(series: pd.Series) -> bool:
    """Return True if the series has 0 or 1 unique non-null values."""
    return series.dropna().nunique() <= 1


def is_near_constant(series: pd.Series, threshold: float = 0.99) -> bool:
    """
    Return True if the most frequent value accounts for more than the threshold.
    Example: threshold=0.99 means 99% of values are identical.
    """
    if series.dropna().empty:
        return True
    return series.value_counts(normalize=True).iloc[0] > threshold


# -----------------------------
# Master Detection Function
# -----------------------------

def detect_feature_types(df: pd.DataFrame, config=None):
    config = config or {}
    low_card = config.get("low_cardinality_threshold", 20)

    feature_types = {
        "numeric": [],
        "binary": [],
        "categorical": [],
        "high_cardinality": [],
        "datetime": [],
    }

    for col in df.columns:
        series = df[col]

        # 1. Already datetime dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            feature_types["datetime"].append(col)
            continue

        # 2. Try parsing strings as datetime (ONLY for object dtype)
        if pd.api.types.is_object_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                parsed = pd.to_datetime(non_null, errors="coerce")
                parse_rate = parsed.notna().mean()

                if parse_rate >= 0.8:
                    feature_types["datetime"].append(col)
                    continue

        # 3. Numeric detection
        if pd.api.types.is_numeric_dtype(series):
            if series.dropna().nunique() == 2:
                feature_types["binary"].append(col)
            else:
                feature_types["numeric"].append(col)
            continue

        # 4. Categorical vs high-cardinality
        nunique = series.dropna().nunique()
        if nunique <= low_card:
            feature_types["categorical"].append(col)
        else:
            feature_types["high_cardinality"].append(col)

    return feature_types