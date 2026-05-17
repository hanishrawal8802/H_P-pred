"""Model trainer: trains LR, RF, XGBoost with MLflow tracking and auto-selection."""
import os
import json
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier

from src.utils.logger import get_logger
from src.utils.config_loader import cfg
from src.ingestion.data_loader import DataLoader
from src.preprocessing.preprocessor import DataPreprocessor
from src.features.feature_engineer import FeatureEngineer

log = get_logger("trainer")


class ModelTrainer:
    """
    Orchestrates training of multiple models, logs to MLflow,
    and auto-selects the best model by ROC-AUC.
    """

    MODEL_CLASSES = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
    }

    def __init__(self):
        self.experiment_name = cfg.get("training.experiment_name", "telco-churn-experiment")
        self.metric_for_best = cfg.get("training.metric_for_best_model", "roc_auc")
        self.test_size = float(cfg.get("data.test_size", 0.2))
        self.random_state = int(cfg.get("data.random_state", 42))
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.model_configs = cfg.get_section("training").get("models", {})
        self.best_model = None
        self.best_run_id: Optional[str] = None
        self.best_metrics: Dict = {}

    def _setup_mlflow(self) -> None:
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            log.info(f"MLflow tracking URI: {self.tracking_uri}")
            log.info(f"Experiment: {self.experiment_name}")
        except Exception as e:
            log.warning(f"MLflow setup failed, continuing without MLflow: {e}")
            self.tracking_uri = None

    def _get_model(self, name: str) -> Any:
        """Instantiate model with default params."""
        if name == "logistic_regression":
            return LogisticRegression(
                max_iter=500, class_weight="balanced", random_state=self.random_state
            )
        elif name == "random_forest":
            return RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=self.random_state, n_jobs=-1
            )
        elif name == "xgboost":
            return XGBClassifier(
                n_estimators=100,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
            )
        raise ValueError(f"Unknown model: {name}")

    def _get_param_grid(self, name: str) -> Dict:
        """Build hyperparameter grid from config."""
        model_cfg = self.model_configs.get(name, {})
        return model_cfg.get("hyperparams", {})

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        }

    def _train_single(
        self,
        name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[Any, Dict, str]:
        """Train one model with hyperparameter search + MLflow logging."""
        log.info(f"Training: {name}")
        model = self._get_model(name)
        param_grid = self._get_param_grid(name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{name}_{timestamp}"
        run_id = "local"

        if self.tracking_uri:
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id

                # Hyperparameter search
                if param_grid:
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                    search = RandomizedSearchCV(
                        model, param_grid,
                        n_iter=10,
                        cv=cv,
                        scoring="roc_auc",
                        n_jobs=-1,
                        random_state=self.random_state,
                        verbose=0,
                    )
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    mlflow.log_params(search.best_params_)
                    log.info(f"Best params ({name}): {search.best_params_}")
                else:
                    model.fit(X_train, y_train)
                    mlflow.log_params({"model": name})

                # Evaluate
                y_pred = model.predict(X_test)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = model.decision_function(X_test)

                metrics = self._compute_metrics(y_test, y_pred, y_prob)

                # Log to MLflow
                mlflow.log_metrics(metrics)
                mlflow.log_param("model_type", name)
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))

                # Log report as artifact
                report = classification_report(y_test, y_pred)
                report_path = self.models_dir / f"{name}_report.txt"
                report_path.write_text(report)
                mlflow.log_artifact(str(report_path))

                # Log model
                if name == "xgboost":
                    mlflow.xgboost.log_model(model, artifact_path="model")
                else:
                    mlflow.sklearn.log_model(model, artifact_path="model")

                log.info(
                    f"{name} → Accuracy:{metrics['accuracy']} "
                    f"F1:{metrics['f1_score']} ROC-AUC:{metrics['roc_auc']}"
                )
        else:
            # Train without MLflow
            if param_grid:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                search = RandomizedSearchCV(
                    model, param_grid,
                    n_iter=10,
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=0,
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                log.info(f"Best params ({name}): {search.best_params_}")
            else:
                model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)

            metrics = self._compute_metrics(y_test, y_pred, y_prob)

            # Save report locally
            report = classification_report(y_test, y_pred)
            report_path = self.models_dir / f"{name}_report.txt"
            report_path.write_text(report)

            log.info(
                f"{name} → Accuracy:{metrics['accuracy']} "
                f"F1:{metrics['f1_score']} ROC-AUC:{metrics['roc_auc']}"
            )

        return model, metrics, run_id

    def run(self) -> Tuple[Any, str, Dict]:
        """Full training pipeline: prep data → train all models → select best."""
        self._setup_mlflow()

        # Run preprocessing pipeline
        log.info("Running full preprocessing pipeline...")
        DataLoader().run()
        DataPreprocessor().run()
        fe = FeatureEngineer()
        X, y = fe.run()

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state, stratify=y
        )
        log.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

        results = {}
        for name, model_cfg in self.model_configs.items():
            if not model_cfg.get("enabled", True):
                log.info(f"Skipping {name} (disabled in config)")
                continue
            try:
                model, metrics, run_id = self._train_single(
                    name, X_train, X_test, y_train, y_test
                )
                results[name] = {"model": model, "metrics": metrics, "run_id": run_id}
            except Exception as exc:
                log.error(f"Failed to train {name}: {exc}")

        if not results:
            raise RuntimeError("No models were successfully trained.")

        # Select best by configured metric
        best_name = max(results, key=lambda n: results[n]["metrics"][self.metric_for_best])
        self.best_model = results[best_name]["model"]
        self.best_run_id = results[best_name]["run_id"]
        self.best_metrics = results[best_name]["metrics"]

        log.info(
            f"Best model: {best_name} | "
            f"{self.metric_for_best}: {self.best_metrics[self.metric_for_best]}"
        )

        # Save best model locally
        best_path = self.models_dir / "best_model.pkl"
        joblib.dump(self.best_model, best_path)
        meta = {
            "model_name": best_name,
            "run_id": self.best_run_id,
            "metrics": self.best_metrics,
            "all_results": {
                n: r["metrics"] for n, r in results.items()
            },
        }
        (self.models_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2))
        log.info(f"Best model saved: {best_path}")
        return self.best_model, self.best_run_id, self.best_metrics


def main():
    trainer = ModelTrainer()
    model, run_id, metrics = trainer.run()
    print(f"\n✅ Training complete!")
    print(f"Run ID: {run_id}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
