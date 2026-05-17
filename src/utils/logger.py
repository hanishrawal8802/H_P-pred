"""Structured logging utility using loguru."""
import sys
import os
from loguru import logger
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs") -> "logger":
    """
    Configure and return a loguru logger instance.

    Args:
        name: Logger name (used in log file name).
        log_dir: Directory to store log files.

    Returns:
        Configured loguru logger.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
    )

    # File handler (rotating)
    logger.add(
        log_path / f"{name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=log_level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        enqueue=True,
    )

    return logger.bind(module=name)


# Default project logger
log = get_logger("mlops_churn")
