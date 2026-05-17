"""MLflow model loader for inference — loads Production model from registry."""
import os
import mlflow
import mlflow.pyfunc
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from src.utils.logger import get_logger
from src.utils.config_loader import cfg

log = get_logger("predictor")


class ChurnPredictor:
    """
    Loads the Production model from MLflow Registry (or local fallback)
    and exposes a predict() method for the FastAPI layer.
    """

    COLUMN_MAPPING = {
        "gender": "Gender",
        "senior_citizen": "Senior Citizen",
        "partner": "Partner",
        "dependents": "Dependents",
        "number_of_dependents": "Number of Dependents",
        "number_of_referrals": "Number of Referrals",
        "phone_service": "Phone Service",
        "multiple_lines": "Multiple Lines",
        "internet_service": "Internet Service",
        "online_security": "Online Security",
        "online_backup": "Online Backup",
        "device_protection": "Device Protection",
        "tech_support": "Tech Support",
        "streaming_tv": "Streaming TV",
        "streaming_movies": "Streaming Movies",
        "contract": "Contract",
        "paperless_billing": "Paperless Billing",
        "payment_method": "Payment Method",
        "tenure_months": "Tenure Months",
        "monthly_charges": "Monthly Charges",
        "total_charges": "Total Charges",
        "total_long_distance_charges": "Total Long Distance Charges",
        "total_revenue": "Total Revenue",
        "avg_monthly_long_distance_charges": "Avg Monthly Long Distance Charges",
        "avg_monthly_gb_download": "Avg Monthly GB Download",
        "cltv": "CLTV",
        "city": "City",
        "state": "State",
        "country": "Country",
    }

    def __init__(self):
        self.model_name = cfg.get("mlflow.model_name", "churn-classifier")
        self.model_stage = os.getenv("MODEL_STAGE", "Production")
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.feature_pipeline_path = cfg.get("features.pipeline_path", "models/feature_pipeline.pkl")
        self.local_model_path = "models/best_model.pkl"

        self.model = None
        self.feature_pipeline = None
        self.model_version: Optional[str] = None
        self.is_loaded: bool = False

    def _load_feature_pipeline(self) -> None:
        """Load serialised sklearn feature pipeline."""
        path = Path(self.feature_pipeline_path)
        if path.exists():
            self.feature_pipeline = joblib.load(path)
            log.info(f"Feature pipeline loaded from: {path}")
        else:
            log.warning(f"Feature pipeline not found at {path}. Raw features will be used.")

    def _load_mlflow_model(self) -> bool:
        """Attempt to load model from MLflow registry."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            model_uri = f"models:/{self.model_name}/{self.model_stage}"
            self.model = mlflow.pyfunc.load_model(model_uri)
            log.info(f"Loaded model from MLflow: {model_uri}")
            # Get version
            from mlflow.tracking import MlflowClient
            client = MlflowClient(self.tracking_uri)
            versions = client.get_latest_versions(self.model_name, stages=[self.model_stage])
            if versions:
                self.model_version = versions[0].version
            return True
        except Exception as exc:
            log.warning(f"Could not load from MLflow ({exc}). Falling back to local model.")
            return False

    def _load_local_model(self) -> bool:
        """Load locally saved model as fallback."""
        path = Path(self.local_model_path)
        if path.exists():
            self.model = joblib.load(path)
            self.model_version = "local"
            log.info(f"Loaded local model from: {path}")
            return True
        log.error(f"No local model found at {path}")
        return False

    def load(self) -> None:
        """Load model and feature pipeline."""
        self._load_feature_pipeline()
        loaded = self._load_mlflow_model()
        if not loaded:
            loaded = self._load_local_model()
        if not loaded:
            raise RuntimeError("No model available. Train a model first.")
        self.is_loaded = True
        log.info(f"Predictor ready. Model version: {self.model_version}")

    def _build_dataframe(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Convert API feature dict to model-compatible DataFrame."""
        renamed = {}
        for api_key, value in features.items():
            col_name = self.COLUMN_MAPPING.get(api_key, api_key)
            renamed[col_name] = value
        return pd.DataFrame([renamed])

    def _get_confidence(self, probability: float) -> str:
        if probability >= 0.80:
            return "High"
        elif probability >= 0.60:
            return "Medium"
        return "Low"

    def predict(self, features: Dict[str, Any]) -> Tuple[str, float, str]:
        """
        Run a single prediction.

        Returns:
            (prediction_label, probability, confidence)
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        df = self._build_dataframe(features)

        # Apply feature pipeline if available
        if self.feature_pipeline is not None:
            try:
                X = self.feature_pipeline.transform(df)
            except Exception as exc:
                log.warning(f"Feature pipeline transform failed: {exc}. Using raw features.")
                X = df
        else:
            X = df

        # Predict
        try:
            if hasattr(self.model, "predict_proba"):
                prob = float(self.model.predict_proba(X)[0][1])
            elif hasattr(self.model, "predict"):
                # MLflow pyfunc model
                result = self.model.predict(pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X)
                prob = float(result[0]) if len(result) > 0 else 0.5
            else:
                prob = 0.5
        except Exception as exc:
            log.error(f"Prediction failed: {exc}")
            raise

        label = "Churn" if prob >= 0.5 else "No Churn"
        confidence = self._get_confidence(prob)

        return label, prob, confidence


# Singleton instance for FastAPI
predictor = ChurnPredictor()
