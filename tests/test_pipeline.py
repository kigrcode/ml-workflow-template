"""
Tests for src/features/pipeline.py

Verifies that FeatureEngineeringPipeline fits and transforms
correctly on synthetic data covering all feature types.
"""

import pandas as pd
import numpy as np
import pytest

from src.features.pipeline import FeatureEngineeringPipeline


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def sample_df():
    """
    Synthetic dataframe covering all feature types.
    Includes missing values to test imputation.
    """
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        "numeric_1":   np.random.randn(n),
        "numeric_2":   np.random.randn(n) * 10 + 5,
        "binary_col":  np.random.choice([0, 1], n),
        "cat_col":     np.random.choice(["a", "b", "c"], n),
        "high_card":   [f"id_{i % 60}" for i in range(n)],
    })

    # Introduce missing values
    df.loc[df.sample(20, random_state=1).index, "numeric_1"] = np.nan
    df.loc[df.sample(10, random_state=2).index, "cat_col"] = np.nan

    return df


@pytest.fixture
def default_config():
    return {
        "scaler": "standard",
        "hashing_dim": 8,
        "feature_selection": {
            "variance_threshold": 0.0,
            "correlation_threshold": None,
            "protected_features": []
        }
    }


# =========================================================
# TESTS — fit and transform
# =========================================================

def test_pipeline_fits_without_error(sample_df, default_config):
    pipeline = FeatureEngineeringPipeline(config=default_config)
    pipeline.fit(sample_df)
    assert pipeline.fitted


def test_pipeline_transform_produces_dataframe(sample_df, default_config):
    pipeline = FeatureEngineeringPipeline(config=default_config)
    pipeline.fit(sample_df)
    result = pipeline.transform(sample_df)
    assert isinstance(result, pd.DataFrame)


def test_pipeline_no_nans_after_transform(sample_df, default_config):
    pipeline = FeatureEngineeringPipeline(config=default_config)
    pipeline.fit(sample_df)
    result = pipeline.transform(sample_df)
    assert result.isna().sum().sum() == 0


def test_pipeline_no_infinite_values(sample_df, default_config):
    pipeline = FeatureEngineeringPipeline(config=default_config)
    pipeline.fit(sample_df)
    result = pipeline.transform(sample_df)
    assert not np.isinf(result.to_numpy()).any()


def test_pipeline_all_numeric_output(sample_df, default_config):
    pipeline = FeatureEngineeringPipeline(config=default_config)
    pipeline.fit(sample_df)
    result = pipeline.transform(sample_df)
    assert all(pd.api.types.is_numeric_dtype(result[col]) for col in result.columns)


def test_pipeline_transform_before_fit_raises(sample_df, default_config):
    pipeline = FeatureEngineeringPipeline(config=default_config)
    with pytest.raises(RuntimeError):
        pipeline.transform(sample_df)


def test_pipeline_fit_transform_matches_fit_then_transform(sample_df, default_config):
    pipeline_1 = FeatureEngineeringPipeline(config=default_config)
    result_1 = pipeline_1.fit_transform(sample_df)

    pipeline_2 = FeatureEngineeringPipeline(config=default_config)
    pipeline_2.fit(sample_df)
    result_2 = pipeline_2.transform(sample_df)

    pd.testing.assert_frame_equal(result_1, result_2)


def test_pipeline_scalers(sample_df):
    for scaler in ["standard", "minmax", "robust"]:
        config = {"scaler": scaler, "hashing_dim": 8}
        pipeline = FeatureEngineeringPipeline(config=config)
        pipeline.fit(sample_df)
        result = pipeline.transform(sample_df)
        assert result.isna().sum().sum() == 0