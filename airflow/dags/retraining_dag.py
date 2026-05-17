"""
Airflow DAG: Automatic Retraining Pipeline
Triggered by drift detection or manually via Airflow UI.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator


default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def task_retrain(**context):
    import json
    from src.training.trainer import ModelTrainer
    trainer = ModelTrainer()
    model, run_id, metrics = trainer.run()
    context["ti"].xcom_push(key="run_id", value=run_id)
    context["ti"].xcom_push(key="metrics", value=json.dumps(metrics))
    return f"Retraining complete. Run ID: {run_id}"


def task_register_retrained(**context):
    import json
    from src.training.registry import ModelRegistry
    run_id = context["ti"].xcom_pull(key="run_id", task_ids="retrain")
    metrics_str = context["ti"].xcom_pull(key="metrics", task_ids="retrain")
    metrics = json.loads(metrics_str)
    registry = ModelRegistry()
    promoted = registry.run(run_id, metrics)
    return f"New model promoted: {promoted}"


def task_notify(**context):
    """Log retraining completion (extend with email/Slack as needed)."""
    run_id = context["ti"].xcom_pull(key="run_id", task_ids="retrain")
    print(f"✅ Retraining pipeline completed. New model run: {run_id}")
    return "Notification sent."


with DAG(
    dag_id="churn_retraining_pipeline",
    default_args=default_args,
    description="Auto-retraining DAG triggered by drift detection",
    schedule=None,   # triggered externally
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "churn", "retraining"],
) as dag:

    start = EmptyOperator(task_id="start")
    retrain = PythonOperator(task_id="retrain", python_callable=task_retrain)
    register = PythonOperator(task_id="register_retrained", python_callable=task_register_retrained)
    notify = PythonOperator(task_id="notify", python_callable=task_notify)
    end = EmptyOperator(task_id="end")

    start >> retrain >> register >> notify >> end
