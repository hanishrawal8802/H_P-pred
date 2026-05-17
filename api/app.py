"""FastAPI application — Telecom Churn Prediction Service."""
import time
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
