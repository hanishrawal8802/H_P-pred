"""Unit tests for the DataPreprocessor module."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.preprocessor import DataPreprocessor


@pytest.fixture
def sample_df():
    """Create a minimal sample DataFrame mimicking the IBM Telco dataset."""
    return pd.DataFrame({
        "CustomerID": ["001", "002", "003", "004", "005"],
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "Senior Citizen": ["No", "Yes", "No", "No", "Yes"],
        "Partner": ["Yes", "No", "Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No", "No"],
        "Tenure Months": [12, 24, 6, 48, 1],
        "Monthly Charges": [65.5, 89.0, 45.0, 110.0, 20.0],
        "Total Charges": ["786.0", "2136.0", "270.0", "5280.0", "  "],
        "Internet Service": ["Fiber optic", "DSL", "No", "Fiber optic", "DSL"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "Month-to-month"],
        "Churn Label": ["Yes", "No", "No", "Yes", "Yes"],
        "Lat Long": ["34.05, -118.24"] * 5,
        "CustomerID": ["001", "002", "003", "004", "005"],
    })


@pytest.fixture
def preprocessor():
    return DataPreprocessor()


class TestDataPreprocessor:

    def test_drop_unnecessary_columns(self, preprocessor, sample_df):
        result = preprocessor.drop_unnecessary_columns(sample_df.copy())
        assert "Lat Long" not in result.columns
        assert "CustomerID" not in result.columns

    def test_fix_data_types_total_charges(self, preprocessor, sample_df):
        result = preprocessor.fix_data_types(sample_df.copy())
        assert pd.api.types.is_float_dtype(result["Total Charges"])

    def test_fix_data_types_whitespace_becomes_null(self, preprocessor, sample_df):
        result = preprocessor.fix_data_types(sample_df.copy())
        # Row with "  " should become NaN
        assert result["Total Charges"].isnull().sum() >= 1

    def test_encode_target(self, preprocessor, sample_df):
        result = preprocessor.encode_target(sample_df.copy())
        assert set(result["Churn Label"].unique()).issubset({0, 1})
        assert result["Churn Label"].dtype in [np.int64, np.int32, int]

    def test_handle_missing_values_no_nulls_remain(self, preprocessor, sample_df):
        df = preprocessor.fix_data_types(sample_df.copy())
        df = preprocessor.encode_target(df)
        result = preprocessor.handle_missing_values(df)
        assert result["Total Charges"].isnull().sum() == 0

    def test_remove_duplicates(self, preprocessor, sample_df):
        df_with_dup = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
        result = preprocessor.remove_duplicates(df_with_dup)
        assert len(result) == len(sample_df)

    def test_encode_target_raises_on_missing_column(self, preprocessor, sample_df):
        df = sample_df.drop(columns=["Churn Label"])
        with pytest.raises(ValueError, match="Target column"):
            preprocessor.encode_target(df)
