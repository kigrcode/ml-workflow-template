import pandas as pd
import numpy as np


# --------------------------------------------------------
# Basic Type Checks
# --------------------------------------------------------

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
    """Return True if series is datetime or parseable as datetime."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if not pd.api.types.is_object_dtype(series):
        return False

    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    parsed = pd.to_datetime(non_null, errors="coerce")
    parse_rate = parsed.notna().mean()

    return parse_rate >= 0.8


# --------------------------------------------------------
# Cardinality Checks
# --------------------------------------------------------

def is_high_cardinality(series: pd.Series, threshold: int = 50) -> bool:
    """Return True if the number of unique values exceeds the threshold."""
    return series.nunique(dropna=True) > threshold


def is_low_cardinality(series: pd.Series, threshold: int = 20) -> bool:
    """Return True if the number of unique values is below or equal to the threshold."""
    return series.nunique(dropna=True) <= threshold


# --------------------------------------------------------
# Missingness Checks
# --------------------------------------------------------

def missing_ratio(series: pd.Series) -> float:
    """Return the proportion of missing values in the series."""
    return series.isna().mean()


def is_missing_heavy(series: pd.Series, threshold: float = 0.4) -> bool:
    """Return True if missingness exceeds the threshold."""
    return missing_ratio(series) > threshold


# --------------------------------------------------------
# Constant / Near-Constant
# --------------------------------------------------------

def is_constant(series: pd.Series) -> bool:
    """Return True if the series has 0 or 1 unique non-null values."""
    return series.dropna().nunique() <= 1


def is_near_constant(series: pd.Series, threshold: float = 0.99) -> bool:
    """
    Return True if the most frequent value accounts for more than the threshold.
    """
    if series.dropna().empty:
        return True
    return series.value_counts(normalize=True).iloc[0] > threshold


# --------------------------------------------------------
# Ordinality Detection
# --------------------------------------------------------

def is_ordinal(series: pd.Series, target: pd.Series) -> bool:
    """
    Detect if a low cardinality numeric feature is ordinal by checking
    if the mean target value increases or decreases monotonically
    across the feature values.

    Only applies to numeric features — string categoricals are always
    treated as nominal.

    Returns True if the relationship is monotonically increasing or
    decreasing, False otherwise.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False

    unique_vals = sorted(series.dropna().unique())

    if len(unique_vals) < 3:
        return False

    mean_targets = [
        target[series == v].mean()
        for v in unique_vals
        if len(target[series == v]) > 0
    ]

    if len(mean_targets) < 3:
        return False

    increasing = all(x <= y for x, y in zip(mean_targets, mean_targets[1:]))
    decreasing = all(x >= y for x, y in zip(mean_targets, mean_targets[1:]))

    return increasing or decreasing


# --------------------------------------------------------
# Master Detection Function
# --------------------------------------------------------

def detect_feature_types(df: pd.DataFrame, config=None, target: pd.Series = None):
    """
    Detect feature types for all columns in the dataframe.

    If target is provided, automatically detects ordinal features
    using monotonicity test. Config overrides take precedence.

    Returns dict with keys:
        numeric, binary, categorical, high_cardinality, datetime,
        ordinal (new)
    """
    config = config or {}
    low_card = config.get("low_cardinality_threshold", 20)

    # Config overrides
    overrides = config.get("feature_type_overrides", {})
    force_ordinal = set(overrides.get("ordinal", []))
    force_nominal = set(overrides.get("nominal", []))

    feature_types = {
        "numeric": [],
        "binary": [],
        "categorical": [],
        "high_cardinality": [],
        "datetime": [],
        "ordinal": [],
    }

    for col in df.columns:
        series = df[col]

        # 1. Config override — force ordinal
        if col in force_ordinal:
            feature_types["ordinal"].append(col)
            continue

        # 2. Config override — force nominal/categorical
        if col in force_nominal:
            feature_types["categorical"].append(col)
            continue

        # 3. Already datetime dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            feature_types["datetime"].append(col)
            continue

        # 4. Try parsing strings as datetime (ONLY for object dtype)
        if pd.api.types.is_object_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                parsed = pd.to_datetime(non_null, errors="coerce")
                parse_rate = parsed.notna().mean()
                if parse_rate >= 0.8:
                    feature_types["datetime"].append(col)
                    continue

        # 5. Numeric detection
        if pd.api.types.is_numeric_dtype(series):
            n_unique = series.dropna().nunique()

            # Binary
            if n_unique == 2:
                feature_types["binary"].append(col)
                continue

            # Low cardinality numeric — check for ordinality
            if n_unique <= low_card:
                if target is not None and is_ordinal(series, target):
                    feature_types["ordinal"].append(col)
                else:
                    # Without target or non-monotonic — treat as categorical
                    feature_types["categorical"].append(col)
                continue

            # High cardinality numeric
            feature_types["numeric"].append(col)
            continue

        # 6. String/categorical — always nominal
        nunique = series.dropna().nunique()
        if nunique <= low_card:
            feature_types["categorical"].append(col)
        else:
            feature_types["high_cardinality"].append(col)

    return feature_types