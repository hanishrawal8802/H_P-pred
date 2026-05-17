"""Model evaluation: generates confusion matrix, ROC curve, feature importance."""
import json
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Any

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    ConfusionMatrixDisplay,
)

from src.utils.logger import get_logger
from src.utils.config_loader import cfg

log = get_logger("evaluator")


class ModelEvaluator:
    """
    Generates evaluation artifacts for a trained model:
      - Confusion matrix plot
      - ROC curve plot
      - Feature importance (if available)
      - JSON metrics summary
    Logs all artifacts to the active MLflow run.
    """

    def __init__(self, output_dir: str = "models/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
        run_id: Optional[str] = None,
    ) -> str:
        fig, ax = plt.subplots(figsize=(7, 6))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
        disp.plot(ax=ax, colorbar=True, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = self.output_dir / f"confusion_matrix_{model_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Confusion matrix saved: {path}")
        return str(path)

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "model",
    ) -> str:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#4e79a7", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random Classifier")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#4e79a7")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve — {model_name}", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        path = self.output_dir / f"roc_curve_{model_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"ROC curve saved: {path}")
        return str(path)

    def plot_feature_importance(
        self,
        model: Any,
        feature_names: list,
        model_name: str = "model",
        top_n: int = 20,
    ) -> Optional[str]:
        """Plot feature importance for tree-based models."""
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_[0])
            else:
                log.warning(f"Model {model_name} does not support feature importance.")
                return None

            if len(feature_names) != len(importances):
                log.warning("Feature names / importance length mismatch. Skipping plot.")
                return None

            feat_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances,
            }).sort_values("importance", ascending=False).head(top_n)

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=feat_df, x="importance", y="feature", ax=ax, palette="viridis")
            ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            path = self.output_dir / f"feature_importance_{model_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info(f"Feature importance plot saved: {path}")
            return str(path)
        except Exception as exc:
            log.warning(f"Could not generate feature importance: {exc}")
            return None

    def save_metrics_json(self, metrics: Dict, model_name: str) -> str:
        path = self.output_dir / f"metrics_{model_name}.json"
        path.write_text(json.dumps(metrics, indent=2))
        log.info(f"Metrics JSON saved: {path}")
        return str(path)

    def log_to_mlflow(self, run_id: str, artifacts: list) -> None:
        """Log all generated artifact paths to an existing MLflow run."""
        try:
            with mlflow.start_run(run_id=run_id):
                for artifact_path in artifacts:
                    if artifact_path and Path(artifact_path).exists():
                        mlflow.log_artifact(artifact_path, artifact_path="evaluation")
            log.info(f"Artifacts logged to MLflow run: {run_id}")
        except Exception as exc:
            log.warning(f"Could not log to MLflow: {exc}")

    def run(
        self,
        model: Any,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        metrics: Dict,
        model_name: str = "model",
        run_id: Optional[str] = None,
        feature_names: Optional[list] = None,
    ) -> Dict:
        log.info(f"Generating evaluation artifacts for: {model_name}")

        artifacts = []
        artifacts.append(self.plot_confusion_matrix(y_true, y_pred, model_name))
        artifacts.append(self.plot_roc_curve(y_true, y_prob, model_name))
        artifacts.append(self.save_metrics_json(metrics, model_name))

        if feature_names:
            fi_path = self.plot_feature_importance(model, feature_names, model_name)
            if fi_path:
                artifacts.append(fi_path)

        if run_id:
            self.log_to_mlflow(run_id, artifacts)

        return {"artifacts": artifacts, "metrics": metrics}
