"""Unit tests for FeatureEngineer."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.feature_engineer import FeatureEngineer


@pytest.fixture
def sample_processed_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "Tenure Months": np.random.randint(1, 72, n),
        "Monthly Charges": np.random.uniform(20, 120, n),
        "Total Charges": np.random.uniform(100, 5000, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Senior Citizen": np.random.choice([0, 1], n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "Internet Service": np.random.choice(["Fiber optic", "DSL", "No"], n),
        "Churn Label": np.random.choice([0, 1], n),
    })


@pytest.fixture
def feature_engineer():
    fe = FeatureEngineer()
    fe.k_features = 5  # small for testing
    return fe


class TestFeatureEngineer:

    def test_identify_columns(self, feature_engineer, sample_processed_df):
        num_cols, cat_cols = feature_engineer.identify_columns(sample_processed_df)
        assert "Tenure Months" in num_cols
        assert "Gender" in cat_cols
        assert "Churn Label" not in num_cols
        assert "Churn Label" not in cat_cols

    def test_build_pipeline_returns_pipeline(self, feature_engineer, sample_processed_df):
        num_cols, cat_cols = feature_engineer.identify_columns(sample_processed_df)
        pipeline = feature_engineer.build_pipeline(num_cols, cat_cols)
        assert hasattr(pipeline, "fit_transform")

    def test_fit_transform_shape(self, feature_engineer, sample_processed_df):
        X, y, _, _ = feature_engineer.fit_transform(sample_processed_df)
        assert X.shape[0] == len(sample_processed_df)
        assert X.shape[1] <= feature_engineer.k_features
        assert len(y) == len(sample_processed_df)

    def test_target_excluded_from_features(self, feature_engineer, sample_processed_df):
        X, y, num_cols, cat_cols = feature_engineer.fit_transform(sample_processed_df)
        assert "Churn Label" not in num_cols
        assert "Churn Label" not in cat_cols
