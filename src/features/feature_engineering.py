"""
Automated, dataset-agnostic feature engineering utilities.
All feature types are opt-in via config.yaml.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from src.features.type_detection import detect_feature_types


def _get_numeric_cols(df: pd.DataFrame) -> list:
    feature_types = detect_feature_types(df)
    return feature_types["numeric"] + feature_types["binary"]


def _get_categorical_cols(df: pd.DataFrame) -> list:
    feature_types = detect_feature_types(df)
    return feature_types["categorical"] + feature_types["high_cardinality"]


# =========================================================
# INTERACTION FEATURES
# =========================================================
def add_interaction_features(
    df: pd.DataFrame,
    max_pairs: int,
) -> pd.DataFrame:
    """
    Multiply pairs of numeric columns to create interaction features.
    max_pairs is set in config.yaml under feature_engineering.max_interaction_pairs
    """
    df = df.copy()
    numeric_cols = _get_numeric_cols(df)
    pairs = list(combinations(numeric_cols, 2))

    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    for col1, col2 in pairs:
        df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

    print(f"Added {len(pairs)} interaction features.")
    return df


# =========================================================
# POLYNOMIAL FEATURES
# =========================================================
def add_polynomial_features(
    df: pd.DataFrame,
    degree: int,
) -> pd.DataFrame:
    """
    Add polynomial features for numeric columns.
    degree is set in config.yaml under feature_engineering.polynomial_degree
    """
    df = df.copy()
    numeric_cols = _get_numeric_cols(df)
    count = 0

    for col in numeric_cols:
        for d in range(2, degree + 1):
            df[f"{col}_pow{d}"] = df[col] ** d
            count += 1

    print(f"Added {count} polynomial features.")
    return df


# =========================================================
# BINNING FEATURES
# =========================================================
def add_binning_features(
    df: pd.DataFrame,
    cols: list,
    n_bins: int,
) -> pd.DataFrame:
    """
    Bin continuous numeric columns into categorical bins.
    cols and n_bins are set in config.yaml under feature_engineering.
    If cols is empty, auto-detects numeric columns.
    """
    df = df.copy()
    numeric_cols = _get_numeric_cols(df)

    if not cols:
        cols = numeric_cols

    count = 0
    for col in cols:
        if col not in df.columns:
            print(f"Skipping {col}: not found in dataframe.")
            continue
        df[f"{col}_binned"] = pd.cut(
            df[col],
            bins=n_bins,
            labels=False,
            duplicates="drop"
        ).astype("float")
        count += 1

    print(f"Added {count} binned features.")
    return df


# =========================================================
# AGGREGATION FEATURES
# =========================================================
def add_aggregation_features(
    df: pd.DataFrame,
    groupby_cols: list,
    agg_cols: list,
) -> pd.DataFrame:
    """
    Add group aggregation features.
    groupby_cols and agg_cols are set in config.yaml under feature_engineering.
    """
    df = df.copy()
    count = 0

    for grp in groupby_cols:
        if grp not in df.columns:
            print(f"Skipping groupby {grp}: not found.")
            continue

        for col in agg_cols:
            if col not in df.columns:
                print(f"Skipping agg {col}: not found.")
                continue

            agg_mean = df.groupby(grp)[col].transform("mean")
            agg_std = df.groupby(grp)[col].transform("std").fillna(0)

            df[f"{grp}_{col}_mean"] = agg_mean
            df[f"{grp}_{col}_std"] = agg_std
            count += 2

    print(f"Added {count} aggregation features.")
    return df


# =========================================================
# ORCHESTRATOR
# =========================================================
def run_feature_engineering(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Run all enabled feature engineering steps based on config.
    Returns enriched dataframe.
    """
    fe_config = config.get("feature_engineering", {})

    if fe_config.get("interactions", False):
        df = add_interaction_features(
            df,
            max_pairs=fe_config.get("max_interaction_pairs", 20)
        )

    if fe_config.get("polynomial", False):
        df = add_polynomial_features(
            df,
            degree=fe_config.get("polynomial_degree", 2)
        )

    if fe_config.get("binning", False):
        df = add_binning_features(
            df,
            cols=fe_config.get("bin_numeric_cols", []),
            n_bins=fe_config.get("n_bins", 5)
        )

    if fe_config.get("aggregations", False):
        df = add_aggregation_features(
            df,
            groupby_cols=fe_config.get("aggregation_groupby", []),
            agg_cols=fe_config.get("aggregation_targets", [])
        )

    return df