"""
Airflow DAG: Full Churn MLOps Pipeline
Runs: ingest → validate → preprocess → features → train → evaluate → register → deploy → drift
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# ─────────────────────────────────────────
# Default args
# ─────────────────────────────────────────
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


# ─────────────────────────────────────────
# Task functions
# ─────────────────────────────────────────
def task_ingest(**context):
    from src.ingestion.data_loader import DataLoader
    loader = DataLoader()
    df = loader.run()
    context["ti"].xcom_push(key="row_count", value=len(df))
    return f"Ingested {len(df)} rows"


def task_validate(**context):
    from src.validation.data_validator import DataValidator
    validator = DataValidator()
    passed, report_path = validator.run()
    context["ti"].xcom_push(key="validation_passed", value=passed)
    context["ti"].xcom_push(key="report_path", value=report_path)
    if not passed:
        raise ValueError(f"Data validation failed. Report: {report_path}")
    return f"Validation passed. Report: {report_path}"


def task_preprocess(**context):
    from src.preprocessing.preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor()
    df = preprocessor.run()
    context["ti"].xcom_push(key="processed_rows", value=len(df))
    return f"Preprocessed {len(df)} rows"


def task_feature_engineer(**context):
    from src.features.feature_engineer import FeatureEngineer
    fe = FeatureEngineer()
    X, y = fe.run()
    context["ti"].xcom_push(key="feature_shape", value=str(X.shape))
    return f"Feature matrix: {X.shape}"


def task_train(**context):
    import json
    from pathlib import Path
    from src.training.trainer import ModelTrainer
    trainer = ModelTrainer()
    model, run_id, metrics = trainer.run()
    context["ti"].xcom_push(key="run_id", value=run_id)
    context["ti"].xcom_push(key="metrics", value=json.dumps(metrics))
    return f"Training complete. Run ID: {run_id}, ROC-AUC: {metrics['roc_auc']}"


def task_evaluate(**context):
    import json
    import numpy as np
    from pathlib import Path
    from src.evaluation.evaluator import ModelEvaluator

    run_id = context["ti"].xcom_pull(key="run_id", task_ids="train")
    metrics_str = context["ti"].xcom_pull(key="metrics", task_ids="train")
    metrics = json.loads(metrics_str)

    evaluator = ModelEvaluator()
    result = evaluator.run(
        model=None,
        y_true=np.array([]),
        y_pred=np.array([]),
        y_prob=np.array([]),
        metrics=metrics,
        model_name="best_model",
        run_id=run_id,
    )
    return f"Evaluation artifacts generated: {result['artifacts']}"


def task_register(**context):
    import json
    from src.training.registry import ModelRegistry

    run_id = context["ti"].xcom_pull(key="run_id", task_ids="train")
    metrics_str = context["ti"].xcom_pull(key="metrics", task_ids="train")
    metrics = json.loads(metrics_str)

    registry = ModelRegistry()
    promoted = registry.run(run_id, metrics)
    context["ti"].xcom_push(key="promoted", value=promoted)
    return f"Model promoted to Production: {promoted}"


def branch_on_promotion(**context):
    """Route based on whether model was promoted."""
    promoted = context["ti"].xcom_pull(key="promoted", task_ids="register")
    if promoted:
        return "deploy"
    return "skip_deploy"


def task_deploy(**context):
    """Restart the API to load the new Production model."""
    import subprocess
    import os
    # Signal the API container to reload (via file-based trigger)
    trigger_path = "/tmp/reload_model"
    with open(trigger_path, "w") as f:
        f.write(datetime.now().isoformat())
    return "Deployment signal sent."


def task_monitor_drift(**context):
    from monitoring.evidently.drift_detector import DriftDetector
    detector = DriftDetector()
    result = detector.run()
    context["ti"].xcom_push(key="drift_detected", value=result.get("dataset_drift", False))
    return f"Drift detection complete. Drift: {result.get('dataset_drift', False)}"


# ─────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────
with DAG(
    dag_id="churn_mlops_pipeline",
    default_args=default_args,
    description="Full MLOps pipeline: ingest → validate → preprocess → train → register → deploy → drift",
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "churn", "telecom"],
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    ingest = PythonOperator(task_id="ingest", python_callable=task_ingest)
    validate = PythonOperator(task_id="validate", python_callable=task_validate)
    preprocess = PythonOperator(task_id="preprocess", python_callable=task_preprocess)
    feature_engineer = PythonOperator(task_id="feature_engineer", python_callable=task_feature_engineer)
    train = PythonOperator(task_id="train", python_callable=task_train)
    evaluate = PythonOperator(task_id="evaluate", python_callable=task_evaluate)
    register = PythonOperator(task_id="register", python_callable=task_register)
    branch = BranchPythonOperator(task_id="branch_promotion", python_callable=branch_on_promotion)
    deploy = PythonOperator(task_id="deploy", python_callable=task_deploy)
    skip_deploy = EmptyOperator(task_id="skip_deploy")
    monitor_drift = PythonOperator(
        task_id="monitor_drift",
        python_callable=task_monitor_drift,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Pipeline flow
    (
        start
        >> ingest
        >> validate
        >> preprocess
        >> feature_engineer
        >> train
        >> evaluate
        >> register
        >> branch
        >> [deploy, skip_deploy]
        >> monitor_drift
        >> end
    )
