"""
Feature engineering utilities.
Extend this per project.
"""

import pandas as pd


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add project-specific features.
    This is a placeholder for real feature engineering.
    """
    df = df.copy()
    # Example:
    # df["income_to_age"] = df["income"] / (df["age"] + 1)
    return df