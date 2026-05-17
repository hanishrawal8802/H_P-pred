"""End-to-end pipeline smoke test (runs full pipeline on sample data)."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture(scope="module")
def temp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("pipeline_test")


@pytest.fixture(scope="module")
def sample_excel(temp_dir):
    """Create a minimal Excel file mimicking the IBM Telco dataset."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "CustomerID": [f"CID-{i:04d}" for i in range(n)],
        "Gender": np.random.choice(["Male", "Female"], n),
        "Senior Citizen": np.random.choice(["Yes", "No"], n),
        "Partner": np.random.choice(["Yes", "No"], n),
        "Dependents": np.random.choice(["Yes", "No"], n),
        "Tenure Months": np.random.randint(1, 72, n),
        "Monthly Charges": np.random.uniform(20.0, 120.0, n).round(2),
        "Total Charges": np.random.uniform(100.0, 5000.0, n).round(2),
        "Internet Service": np.random.choice(["Fiber optic", "DSL", "No"], n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "Payment Method": np.random.choice(["Electronic check", "Mailed check"], n),
        "Paperless Billing": np.random.choice(["Yes", "No"], n),
        "Phone Service": np.random.choice(["Yes", "No"], n),
        "Multiple Lines": np.random.choice(["Yes", "No", "No phone service"], n),
        "Online Security": np.random.choice(["Yes", "No", "No internet service"], n),
        "Online Backup": np.random.choice(["Yes", "No", "No internet service"], n),
        "Device Protection": np.random.choice(["Yes", "No", "No internet service"], n),
        "Tech Support": np.random.choice(["Yes", "No", "No internet service"], n),
        "Streaming TV": np.random.choice(["Yes", "No", "No internet service"], n),
        "Streaming Movies": np.random.choice(["Yes", "No", "No internet service"], n),
        "Churn Label": np.random.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })
    excel_path = temp_dir / "Telco_customer_churn.xlsx"
    df.to_excel(excel_path, index=False, engine="openpyxl")
    return excel_path, df


class TestIngestionPipeline:

    def test_data_loads_from_excel(self, sample_excel, temp_dir, monkeypatch):
        excel_path, original_df = sample_excel
        from src.ingestion.data_loader import DataLoader

        loader = DataLoader()
        monkeypatch.setattr(loader, "source_excel", str(excel_path))
        monkeypatch.setattr(loader, "raw_data_path", str(temp_dir / "raw.csv"))

        df = loader.run()
        assert df is not None
        assert len(df) == len(original_df)
        assert "Churn Label" in df.columns


class TestPreprocessingPipeline:

    def test_preprocessing_runs_without_error(self, sample_excel, temp_dir, monkeypatch):
        excel_path, original_df = sample_excel

        # Save raw CSV first
        raw_path = temp_dir / "raw.csv"
        original_df.to_csv(raw_path, index=False)

        from src.preprocessing.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        monkeypatch.setattr(preprocessor, "raw_path", str(raw_path))
        monkeypatch.setattr(preprocessor, "processed_path", str(temp_dir / "processed.csv"))

        result = preprocessor.run()
        assert result is not None
        assert "Churn Label" in result.columns
        assert set(result["Churn Label"].unique()).issubset({0, 1})

    def test_no_nulls_after_preprocessing(self, sample_excel, temp_dir, monkeypatch):
        excel_path, original_df = sample_excel
        raw_path = temp_dir / "raw.csv"
        original_df.to_csv(raw_path, index=False)

        from src.preprocessing.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        monkeypatch.setattr(preprocessor, "raw_path", str(raw_path))
        monkeypatch.setattr(preprocessor, "processed_path", str(temp_dir / "processed2.csv"))

        result = preprocessor.run()
        assert result.isnull().sum().sum() == 0


class TestFeatureEngineeringPipeline:

    def test_feature_pipeline_produces_array(self, sample_excel, temp_dir, monkeypatch):
        excel_path, original_df = sample_excel
        proc_path = temp_dir / "processed3.csv"

        # Create processed data
        original_df["Churn Label"] = original_df["Churn Label"].map({"Yes": 1, "No": 0})
        original_df = original_df.dropna(subset=["Churn Label"])
        original_df.to_csv(proc_path, index=False)

        from src.features.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()
        fe.k_features = 5
        monkeypatch.setattr(fe, "processed_path", str(proc_path))
        monkeypatch.setattr(fe, "pipeline_path", str(temp_dir / "pipeline.pkl"))

        fe.load_data = lambda: pd.read_csv(proc_path)

        df = pd.read_csv(proc_path)
        X, y, num_cols, cat_cols = fe.fit_transform(df)

        assert X.shape[0] == len(df)
        assert X.shape[1] <= 5
        assert len(y) == len(df)
