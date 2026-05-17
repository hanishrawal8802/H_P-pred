.PHONY: help install lint test train serve docker-up docker-down clean

help:
	@echo "========================================"
	@echo "  MLOps Churn Platform — Commands"
	@echo "========================================"
	@echo "  make install       Install dependencies"
	@echo "  make lint          Run Black + Flake8"
	@echo "  make format        Auto-format with Black"
	@echo "  make test          Run all tests"
	@echo "  make ingest        Run data ingestion"
	@echo "  make train         Train models locally"
	@echo "  make serve         Start FastAPI locally"
	@echo "  make dashboard     Start Streamlit locally"
	@echo "  make docker-up     Start full stack"
	@echo "  make docker-down   Stop full stack"
	@echo "  make clean         Remove generated artifacts"
	@echo "========================================"

install:
	pip install -r requirements.txt
	pip install -e .

lint:
	black --check src/ api/ dashboard/ tests/ airflow/
	flake8 src/ api/ dashboard/ tests/ airflow/ --max-line-length=100 --ignore=E501,W503

format:
	black src/ api/ dashboard/ tests/ airflow/
	isort src/ api/ dashboard/ tests/ airflow/

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-api:
	pytest tests/api/ -v

test-pipeline:
	pytest tests/pipeline/ -v

ingest:
	python -m src.ingestion.data_loader

validate:
	python -m src.validation.data_validator

preprocess:
	python -m src.preprocessing.preprocessor

features:
	python -m src.features.feature_engineer

train:
	python -m src.training.trainer

evaluate:
	python -m src.evaluation.evaluator

drift:
	python -m monitoring.evidently.drift_detector

serve:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/streamlit_app.py --server.port 8501

pipeline: ingest validate preprocess features train evaluate
	@echo "Full pipeline complete."

docker-up:
	docker compose up --build -d
	@echo "Services starting..."
	@echo "MLflow:    http://localhost:5000"
	@echo "Airflow:   http://localhost:8080"
	@echo "API:       http://localhost:8000"
	@echo "Dashboard: http://localhost:8501"
	@echo "Grafana:   http://localhost:3000"
	@echo "Prometheus:http://localhost:9090"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-clean:
	docker compose down -v --remove-orphans

clean:
	rm -rf data/processed/*.csv
	rm -rf data/validation/
	rm -rf models/*.pkl
	rm -rf mlruns/
	rm -rf monitoring/evidently/reports/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

init-airflow:
	docker compose run --rm airflow-webserver airflow db init
	docker compose run --rm airflow-webserver airflow users create \
		--username admin --password admin \
		--firstname Admin --lastname User \
		--role Admin --email admin@example.com
