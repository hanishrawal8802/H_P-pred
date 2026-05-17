# 📡 Production-Grade Telecom Customer Churn MLOps Platform

> End-to-end MLOps system for predicting IBM Telco Customer Churn using XGBoost, MLflow, Airflow, FastAPI, Streamlit, Prometheus, Grafana, and Docker Compose.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Sources                           │
│              Telco_customer_churn.xlsx (33 features)        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Apache Airflow                            │
│   ingest → validate → preprocess → features → train        │
│   → evaluate → register → deploy → drift_detect            │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
  ┌──────────┐    ┌──────────┐    ┌──────────────┐
  │ MLflow   │    │PostgreSQL│    │  Models Dir  │
  │ Tracking │    │ (3 DBs)  │    │  (.pkl files)│
  └──────────┘    └──────────┘    └──────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Service (:8000)                   │
│          /predict  /predict/batch  /health  /metrics        │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
  ┌──────────┐    ┌──────────┐    ┌──────────────┐
  │Streamlit │    │Prometheus│    │   Grafana    │
  │Dashboard │    │  (:9090) │    │   (:3000)    │
  │ (:8501)  │    └──────────┘    └──────────────┘
  └──────────┘
```

---

## 🚀 Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| FastAPI (Docs) | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| Airflow UI | http://localhost:8080 | admin / admin |
| Streamlit Dashboard | http://localhost:8501 | — |
| Grafana | http://localhost:3000 | admin / admin123 |
| Prometheus | http://localhost:9090 | — |

---

## ⚡ Prerequisites

- **Docker Desktop** ≥ 4.x (with 8GB+ RAM allocated)
- **Docker Compose** v2
- **Git**

---

## 🛠️ Setup & Run

### 1. Clone and prepare data

```bash
git clone <your-repo-url>
cd mlops-churn-platform

# Copy your dataset into the data/raw directory
cp /path/to/Telco_customer_churn.xlsx data/raw/
# OR let the ingestion pipeline find it automatically in the project root
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work out of the box)
```

### 3. Start the full stack

```bash
docker compose up --build
```

> First build takes ~5-10 minutes. Subsequent starts are fast.

### 4. Trigger the ML pipeline

Once all services are UP, open **Airflow UI** at http://localhost:8080:
1. Login: `admin` / `admin`
2. Enable the `churn_mlops_pipeline` DAG
3. Click **Trigger DAG** ▶️
4. Watch: ingest → validate → preprocess → train → register → deploy → drift

---

## 🔌 API Usage

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "tenure_months": 12,
    "monthly_charges": 75.50,
    "total_charges": 906.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "paperless_billing": "Yes"
  }'
```

**Response:**
```json
{
  "customer_id": null,
  "prediction": "Churn",
  "probability": 0.823,
  "confidence": "High",
  "model_version": "1",
  "model_name": "churn-classifier"
}
```

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "gender": "Female",
        "tenure_months": 48,
        "monthly_charges": 45.0,
        "contract": "Two year",
    }
)
print(response.json())
```

---

## 🧪 Running Tests

```bash
# Install dependencies locally
pip install -r requirements.txt

# Run all tests
make test

# Unit tests only
make test-unit

# API tests only
make test-api
```

---

## 🔧 Development Commands

```bash
make install       # Install all dependencies
make lint          # Black + Flake8 check
make format        # Auto-format code
make ingest        # Run data ingestion only
make train         # Run full training pipeline
make serve         # Start FastAPI locally (port 8000)
make dashboard     # Start Streamlit locally (port 8501)
make pipeline      # Run full pipeline locally
make docker-up     # Start full Docker stack
make docker-down   # Stop stack
make docker-logs   # Follow all logs
make clean         # Remove generated artifacts
```

---

## 📊 Pipeline Flow

```
Data Ingestion      → Load Excel → Save raw CSV
Data Validation     → Schema checks, null checks, duplicates
Preprocessing       → Type fixing, target encoding, imputation
Feature Engineering → OneHot encoding, StandardScaler, SelectKBest
Model Training      → LR + RandomForest + XGBoost with GridSearch
Experiment Tracking → MLflow: params, metrics, artifacts
Model Registry      → Best model → Staging → Production
Deployment          → FastAPI loads Production model
Monitoring          → Prometheus + Grafana dashboards
Drift Detection     → Evidently AI reports
Auto-Retraining     → Triggered if drift > 15%
```

---

## 📁 Project Structure

```
mlops-churn-platform/
├── config/config.yaml          # Central configuration
├── data/raw/                   # Raw CSV landing zone
├── data/processed/             # Feature-engineered data
├── data/validation/            # Validation HTML reports
├── src/
│   ├── ingestion/             # DataLoader
│   ├── validation/            # DataValidator
│   ├── preprocessing/         # DataPreprocessor
│   ├── features/              # FeatureEngineer
│   ├── training/              # ModelTrainer + ModelRegistry
│   ├── evaluation/            # ModelEvaluator
│   └── utils/                 # Logger, ConfigLoader, DB
├── api/                       # FastAPI app
├── dashboard/                 # Streamlit dashboard
├── airflow/dags/              # Airflow DAGs
├── monitoring/
│   ├── prometheus/            # Prometheus config
│   ├── grafana/               # Grafana dashboards
│   └── evidently/             # Drift detector
├── tests/                     # Unit + API + Pipeline tests
├── models/                    # Serialized models
├── docker/                    # Dockerfiles
├── .github/workflows/         # CI/CD
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## 🌊 Drift Detection

Drift is automatically checked after each pipeline run.  
If **>15%** of features show significant drift (Jensen-Shannon divergence):
1. Alert is logged
2. `churn_retraining_pipeline` DAG is triggered automatically
3. New model is registered and promoted to Production

---

## 🔍 Troubleshooting

| Problem | Solution |
|---------|---------|
| Services not starting | Run `docker compose logs <service>` |
| MLflow connection refused | Wait 60s for MLflow to initialise |
| Model not loaded | Trigger Airflow pipeline first to train |
| Port conflicts | Edit `docker-compose.yml` port mappings |
| Out of memory | Increase Docker Desktop RAM to 8GB+ |
| Airflow DB errors | Run `make init-airflow` |

### Reset everything

```bash
docker compose down -v --remove-orphans
docker volume prune -f
docker compose up --build
```

---

## 🤝 CI/CD

GitHub Actions pipeline (`.github/workflows/ci.yml`):

```
Push to main/develop
    ↓
Lint (Black + Flake8)
    ↓
Unit & API Tests
    ↓
Pipeline Smoke Tests
    ↓
Docker Build (all images)
    ↓ (main branch only)
Push to GitHub Container Registry (GHCR)
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
