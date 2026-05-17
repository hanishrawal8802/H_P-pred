"""MLflow Model Registry: register, promote, and manage model stages."""
import os
import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional, Dict
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config_loader import cfg

log = get_logger("registry")


class ModelRegistry:
    """
    Handles MLflow Model Registry operations:
      - Register a model from a run
      - Promote to Staging → Production
      - Archive old production models
      - Retrieve production model URI
    """

    def __init__(self):
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.model_name = cfg.get("mlflow.model_name", "churn-classifier")
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def register_model(self, run_id: str, artifact_path: str = "model") -> str:
        """Register a model from an MLflow run."""
        model_uri = f"runs:/{run_id}/{artifact_path}"
        log.info(f"Registering model from run {run_id} → '{self.model_name}'")
        try:
            mv = mlflow.register_model(model_uri, self.model_name)
            version = mv.version
            log.info(f"Registered: '{self.model_name}' version {version}")
            return version
        except Exception as exc:
            log.error(f"Model registration failed: {exc}")
            raise

    def promote_to_staging(self, version: str) -> None:
        """Transition a model version to Staging."""
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )
        log.info(f"'{self.model_name}' v{version} → Staging")

    def promote_to_production(self, version: str) -> None:
        """Transition Staging model to Production, archiving previous Production."""
        # Archive current Production versions
        prod_versions = self.client.get_latest_versions(self.model_name, stages=["Production"])
        for pv in prod_versions:
            if pv.version != version:
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=pv.version,
                    stage="Archived",
                )
                log.info(f"Archived '{self.model_name}' v{pv.version}")

        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage="Production",
        )
        log.info(f"'{self.model_name}' v{version} → Production ✅")

    def get_production_model_uri(self) -> Optional[str]:
        """Return the URI of the current Production model."""
        try:
            versions = self.client.get_latest_versions(self.model_name, stages=["Production"])
            if not versions:
                log.warning(f"No Production version found for '{self.model_name}'")
                return None
            uri = f"models:/{self.model_name}/Production"
            log.info(f"Production model URI: {uri}")
            return uri
        except Exception as exc:
            log.error(f"Failed to get production model: {exc}")
            return None

    def get_model_details(self) -> Dict:
        """Return details of all registered model versions."""
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            return {
                v.version: {
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "status": v.status,
                    "created": v.creation_timestamp,
                }
                for v in versions
            }
        except Exception as exc:
            log.error(f"Could not fetch model details: {exc}")
            return {}

    def validate_and_promote(self, version: str, metrics: Dict, threshold: float = 0.80) -> bool:
        """
        Validate model metrics and auto-promote if ROC-AUC exceeds threshold.
        Returns True if promoted.
        """
        roc_auc = metrics.get("roc_auc", 0.0)
        log.info(f"Validating model v{version}: ROC-AUC={roc_auc:.4f} (threshold={threshold})")
        if roc_auc >= threshold:
            self.promote_to_staging(version)
            self.promote_to_production(version)
            log.info(f"Model v{version} promoted to Production.")
            return True
        else:
            log.warning(
                f"Model v{version} did NOT meet threshold "
                f"({roc_auc:.4f} < {threshold}). Not promoted."
            )
            return False

    def run(self, run_id: str, metrics: Dict) -> bool:
        """Full registry pipeline: register → validate → promote."""
        version = self.register_model(run_id)
        promoted = self.validate_and_promote(version, metrics)
        return promoted


def main():
    import json
    meta_path = Path("models/training_metadata.json")
    if not meta_path.exists():
        print("No training metadata found. Run training first.")
        return
    meta = json.loads(meta_path.read_text())
    registry = ModelRegistry()
    registry.run(meta["run_id"], meta["metrics"])


if __name__ == "__main__":
    main()
