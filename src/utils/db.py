"""Database connection and session management using SQLAlchemy."""
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text, Column, String, Float, DateTime, Integer, JSON
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import QueuePool
from datetime import datetime, timezone

from src.utils.logger import get_logger

log = get_logger("db")


def get_db_url() -> str:
    """Construct PostgreSQL URL from environment variables."""
    return (
        f"postgresql+psycopg2://"
        f"{os.getenv('POSTGRES_USER', 'mlops')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'mlops_secret')}@"
        f"{os.getenv('POSTGRES_HOST', 'postgres')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'mlops_db')}"
    )


# ─────────────────────────────────────────
# Engine & Session
# ─────────────────────────────────────────
engine = create_engine(
    get_db_url(),
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────
class PredictionLog(Base):
    """Logs every prediction made by the API."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String, nullable=True)
    prediction = Column(String, nullable=False)
    probability = Column(Float, nullable=False)
    confidence = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    features = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ModelMetrics(Base):
    """Stores evaluation metrics per model run."""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    is_production = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def init_db() -> None:
    """Create all tables if they don't exist."""
    try:
        Base.metadata.create_all(bind=engine)
        log.info("Database tables initialised successfully.")
    except Exception as exc:
        log.error(f"Failed to initialise database: {exc}")
        raise


def check_db_connection() -> bool:
    """Verify database connectivity."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        log.error(f"Database connection failed: {exc}")
        return False


@contextmanager
def get_session() -> Generator:
    """Context manager for database sessions."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
