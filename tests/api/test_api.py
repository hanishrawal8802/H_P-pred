"""API integration tests using FastAPI TestClient."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture
def mock_predictor():
    """Mock the predictor to avoid model loading during tests."""
    with patch("api.app.predictor") as mock:
        mock.is_loaded = True
        mock.model_version = "test-v1"
        mock.model_name = "churn-classifier"
        mock.tracking_uri = "http://localhost:5000"
        mock.predict.return_value = ("Churn", 0.85, "High")
        yield mock


@pytest.fixture
def mock_db():
    """Mock DB session to avoid PostgreSQL dependency."""
    with patch("api.app.get_session") as mock_session, \
         patch("api.app.init_db") as mock_init:
        mock_ctx = MagicMock()
        mock_session.return_value.__enter__ = lambda s: MagicMock()
        mock_session.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_session


@pytest.fixture
def client(mock_predictor, mock_db):
    from api.app import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data


class TestPredictEndpoint:

    def test_predict_returns_200(self, client):
        payload = {
            "gender": "Male",
            "tenure_months": 12,
            "monthly_charges": 75.5,
            "contract": "Month-to-month",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        payload = {"gender": "Female", "monthly_charges": 65.0}
        response = client.post("/predict", json=payload)
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "confidence" in data
        assert "model_version" in data

    def test_predict_probability_in_range(self, client):
        payload = {"tenure_months": 24}
        response = client.post("/predict", json=payload)
        prob = response.json()["probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_batch_returns_200(self, client):
        payload = {
            "customers": [
                {"gender": "Male", "tenure_months": 12},
                {"gender": "Female", "monthly_charges": 90.0},
            ]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert data["total"] == 2

    def test_model_info_endpoint(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "is_loaded" in data
