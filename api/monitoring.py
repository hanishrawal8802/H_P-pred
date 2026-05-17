"""Prometheus metrics for FastAPI monitoring."""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time


# ─────────────────────────────────────────
# Metrics definitions
# ─────────────────────────────────────────
REQUEST_COUNT = Counter(
    "churn_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "churn_api_request_duration_seconds",
    "API request latency",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_COUNTER = Counter(
    "churn_predictions_total",
    "Total churn predictions made",
    ["prediction_label"],
)

PREDICTION_PROBABILITY = Histogram(
    "churn_prediction_probability",
    "Distribution of churn prediction probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

MODEL_LOADED = Gauge(
    "churn_model_loaded",
    "Whether the churn model is currently loaded (1=yes, 0=no)",
)

FAILED_PREDICTIONS = Counter(
    "churn_failed_predictions_total",
    "Total failed prediction requests",
)


# ─────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────
def record_request(method: str, endpoint: str, status: int, duration: float) -> None:
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)


def record_prediction(label: str, probability: float) -> None:
    PREDICTION_COUNTER.labels(prediction_label=label).inc()
    PREDICTION_PROBABILITY.observe(probability)


def metrics_endpoint() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
