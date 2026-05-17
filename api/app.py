"""FastAPI application — Telecom Churn Prediction Service."""
import time
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from api.schemas import (
    CustomerFeatures,
    BatchPredictRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
)
from api.predictor import predictor
from api.monitoring import (
    record_request,
    record_prediction,
    metrics_endpoint,
    MODEL_LOADED,
    FAILED_PREDICTIONS,
)
from src.utils.logger import get_logger
from src.utils.db import init_db, get_session, PredictionLog

log = get_logger("api")


# ─────────────────────────────────────────
# Lifespan: startup / shutdown
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and initialise DB on startup."""
    log.info("Starting Churn Prediction API...")
    try:
        init_db()
        log.info("Database initialised.")
    except Exception as exc:
        log.warning(f"DB init failed (will continue without DB): {exc}")

    try:
        predictor.load()
        MODEL_LOADED.set(1)
        log.info("Model loaded successfully.")
    except Exception as exc:
        MODEL_LOADED.set(0)
        log.error(f"Model load failed: {exc}")

    yield

    log.info("Shutting down API...")
    MODEL_LOADED.set(0)


# ─────────────────────────────────────────
# App
# ─────────────────────────────────────────
app = FastAPI(
    title="Telecom Churn Prediction API",
    description=(
        "Production-grade MLOps API for predicting telecom customer churn. "
        "Built with FastAPI + MLflow + XGBoost."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Middleware: request timing
# ─────────────────────────────────────────
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    record_request(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
        duration=duration,
    )
    response.headers["X-Response-Time"] = f"{duration:.4f}s"
    return response


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["Home"])
async def root():
    """Portal page with links to all services."""
    dashboard_url = os.getenv("DASHBOARD_URL", "http://localhost:8501")
    is_render = os.getenv("IS_RENDER", "false").lower() == "true"
    
    local_services = "" if is_render else """
            <a href="http://localhost:5000" target="_blank" class="card">
                <h2>MLflow Registry <span class="badge">Port 5000</span></h2>
                <p>Experiment tracking, metrics, and model registry.</p>
            </a>
            <a href="http://localhost:8080" target="_blank" class="card">
                <h2>Airflow Pipelines <span class="badge">Port 8080</span></h2>
                <p>Scheduled data ingestion and model retraining DAGs.</p>
            </a>
            <a href="http://localhost:3000" target="_blank" class="card">
                <h2>Grafana Monitoring <span class="badge">Port 3000</span></h2>
                <p>System, API performance, and data drift dashboards.</p>
            </a>
            <a href="http://localhost:9090" target="_blank" class="card">
                <h2>Prometheus Metrics <span class="badge">Port 9090</span></h2>
                <p>Raw time-series metrics collection.</p>
            </a>
    """

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MLOps Churn Platform Portal</title>
        <style>
            body {{ font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; }}
            h1 {{ font-size: 2.5rem; margin-bottom: 0.5rem; font-weight: 700; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
            p {{ color: #94a3b8; margin-bottom: 3rem; font-size: 1.1rem; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; width: 100%; max-width: 1000px; padding: 0 2rem; }}
            .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem; text-decoration: none; color: inherit; transition: all 0.3s ease; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); display: flex; flex-direction: column; }}
            .card:hover {{ transform: translateY(-5px); border-color: #38bdf8; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); }}
            .card h2 {{ margin: 0 0 0.5rem 0; font-size: 1.3rem; color: #f1f5f9; display: flex; align-items: center; justify-content: space-between; }}
            .card p {{ margin: 0; color: #94a3b8; font-size: 0.95rem; }}
            .badge {{ font-size: 0.75rem; background: #0ea5e9; color: white; padding: 0.2rem 0.5rem; border-radius: 999px; font-weight: 600; }}
        </style>
    </head>
    <body>
        <h1>MLOps Churn Platform</h1>
        <p>Your Central Hub for the End-to-End Pipeline {"(Cloud Mode)" if is_render else "(Local Mode)"}</p>
        <div class="grid">
            <a href="{dashboard_url}" target="_blank" class="card">
                <h2>Streamlit Dashboard <span class="badge">{"Web" if is_render else "Port 8501"}</span></h2>
                <p>Interactive UI for predicting churn and viewing insights.</p>
            </a>
            <a href="/docs" target="_blank" class="card">
                <h2>FastAPI Docs <span class="badge">{"Web" if is_render else "Port 8000"}</span></h2>
                <p>Swagger UI for testing prediction endpoints and health checks.</p>
            </a>
            {local_services}
        </div>
    </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        model_version=predictor.model_version,
        model_stage=os.getenv("MODEL_STAGE", "Production"),
    )


@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus scrape endpoint."""
    return metrics_endpoint()


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    customer: CustomerFeatures,
    customer_id: Optional[str] = None,
):
    """
    Predict churn probability for a single customer.

    Returns prediction label, probability, and confidence level.
    """
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not available. Try again later.")

    features = customer.model_dump(exclude_none=False)

    try:
        label, probability, confidence = predictor.predict(features)
    except Exception as exc:
        FAILED_PREDICTIONS.inc()
        log.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")

    record_prediction(label, probability)

    # Log to DB (non-blocking)
    try:
        with get_session() as session:
            session.add(PredictionLog(
                customer_id=customer_id,
                prediction=label,
                probability=probability,
                confidence=confidence,
                model_version=predictor.model_version or "unknown",
                features=features,
            ))
    except Exception as exc:
        log.warning(f"DB logging failed (non-critical): {exc}")

    return PredictionResponse(
        customer_id=customer_id,
        prediction=label,
        probability=round(probability, 4),
        confidence=confidence,
        model_version=predictor.model_version or "unknown",
        model_name=predictor.model_name,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """Batch prediction for multiple customers."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not available.")

    results = []
    for i, customer in enumerate(request.customers):
        features = customer.model_dump(exclude_none=False)
        try:
            label, probability, confidence = predictor.predict(features)
            record_prediction(label, probability)
            results.append(PredictionResponse(
                customer_id=f"batch_{i}",
                prediction=label,
                probability=round(probability, 4),
                confidence=confidence,
                model_version=predictor.model_version or "unknown",
                model_name=predictor.model_name,
            ))
        except Exception as exc:
            FAILED_PREDICTIONS.inc()
            log.warning(f"Batch prediction failed for index {i}: {exc}")

    return BatchPredictionResponse(predictions=results, total=len(results))


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Return information about the currently loaded model."""
    return {
        "model_name": predictor.model_name,
        "model_version": predictor.model_version,
        "model_stage": os.getenv("MODEL_STAGE", "Production"),
        "is_loaded": predictor.is_loaded,
        "tracking_uri": predictor.tracking_uri,
    }


def main():
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("ENVIRONMENT", "development") == "development",
        log_level="info",
    )


if __name__ == "__main__":
    main()
