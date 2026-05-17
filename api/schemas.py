"""Pydantic schemas for FastAPI request/response validation."""
from pydantic import BaseModel, Field
from typing import Optional, List


class CustomerFeatures(BaseModel):
    """Input schema: raw customer features for churn prediction."""

    # Demographics
    gender: Optional[str] = Field(None, example="Male")
    senior_citizen: Optional[str] = Field(None, example="No")
    partner: Optional[str] = Field(None, example="Yes")
    dependents: Optional[str] = Field(None, example="No")
    number_of_dependents: Optional[int] = Field(None, ge=0, example=0)
    number_of_referrals: Optional[int] = Field(None, ge=0, example=2)

    # Services
    phone_service: Optional[str] = Field(None, example="Yes")
    multiple_lines: Optional[str] = Field(None, example="No")
    internet_service: Optional[str] = Field(None, example="Fiber optic")
    online_security: Optional[str] = Field(None, example="No")
    online_backup: Optional[str] = Field(None, example="Yes")
    device_protection: Optional[str] = Field(None, example="No")
    tech_support: Optional[str] = Field(None, example="No")
    streaming_tv: Optional[str] = Field(None, example="Yes")
    streaming_movies: Optional[str] = Field(None, example="No")

    # Contract & Billing
    contract: Optional[str] = Field(None, example="Month-to-month")
    paperless_billing: Optional[str] = Field(None, example="Yes")
    payment_method: Optional[str] = Field(None, example="Electronic check")

    # Financials
    tenure_months: Optional[int] = Field(None, ge=0, le=120, example=12)
    monthly_charges: Optional[float] = Field(None, ge=0, example=75.50)
    total_charges: Optional[float] = Field(None, ge=0, example=906.0)
    total_long_distance_charges: Optional[float] = Field(None, ge=0, example=200.0)
    total_revenue: Optional[float] = Field(None, ge=0, example=1106.0)
    avg_monthly_long_distance_charges: Optional[float] = Field(None, ge=0, example=16.67)
    avg_monthly_gb_download: Optional[float] = Field(None, ge=0, example=25.0)
    cltv: Optional[int] = Field(None, ge=0, example=3500)

    # Location
    city: Optional[str] = Field(None, example="Los Angeles")
    state: Optional[str] = Field(None, example="California")
    country: Optional[str] = Field(None, example="United States")

    model_config = {"json_schema_extra": {"example": {
        "gender": "Male",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 12,
        "monthly_charges": 75.5,
        "total_charges": 906.0,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
    }}}


class BatchPredictRequest(BaseModel):
    customers: List[CustomerFeatures]


class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    prediction: str = Field(..., example="Churn")
    probability: float = Field(..., ge=0.0, le=1.0, example=0.823)
    confidence: str = Field(..., example="High")
    model_version: str = Field(..., example="1")
    model_name: str = Field(..., example="churn-classifier")

    model_config = {"json_schema_extra": {"example": {
        "customer_id": "1234-ABCDE",
        "prediction": "Churn",
        "probability": 0.823,
        "confidence": "High",
        "model_version": "1",
        "model_name": "churn-classifier",
    }}}


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    model_version: Optional[str] = None
    model_stage: str = "Production"
