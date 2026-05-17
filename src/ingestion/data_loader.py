"""Data ingestion module: loads IBM Telco Churn Excel → CSV."""
import os
import shutil
import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger
from src.utils.config_loader import cfg

log = get_logger("ingestion")


class DataLoader:
    """
    Handles loading raw data from the source Excel file and
    persisting it as CSV in the raw data directory.
    """

    def __init__(self):
        self.source_excel: str = cfg.get("data.source_excel", "Telco_customer_churn.xlsx")
        self.raw_data_path: str = cfg.get("data.raw_data_path", "data/raw/telco_churn_raw.csv")
        self.target_column: str = cfg.get("data.target_column", "Churn Label")

    def _resolve_source(self) -> Path:
        """Find source Excel file relative to project root or data dir."""
        candidates = [
            Path(self.source_excel),
            Path("data") / self.source_excel,
            Path("..") / self.source_excel,
            Path(__file__).resolve().parents[3] / self.source_excel,
        ]
        for c in candidates:
            if c.exists():
                log.info(f"Found source file: {c.resolve()}")
                return c
        raise FileNotFoundError(
            f"Source file '{self.source_excel}' not found. "
            f"Searched: {[str(c) for c in candidates]}"
        )

    def load(self) -> pd.DataFrame:
        """Load the Excel file into a DataFrame."""
        source = self._resolve_source()
        log.info(f"Loading data from {source}")
        try:
            df = pd.read_excel(source, engine="openpyxl")
        except Exception as exc:
            log.error(f"Failed to read Excel file: {exc}")
            raise
        log.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        return df

    def validate_columns(self, df: pd.DataFrame) -> None:
        """Basic sanity checks on the raw DataFrame."""
        if df.empty:
            raise ValueError("Loaded DataFrame is empty.")
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found. "
                f"Available: {list(df.columns)}"
            )
        null_pct = df.isnull().mean() * 100
        high_null = null_pct[null_pct > 50]
        if not high_null.empty:
            log.warning(f"Columns with >50% nulls:\n{high_null}")
        log.info(f"Column validation passed. Target distribution:\n{df[self.target_column].value_counts()}")

    def save_raw(self, df: pd.DataFrame) -> str:
        """Save DataFrame to CSV in the raw data directory."""
        out_path = Path(self.raw_data_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info(f"Raw data saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
        return str(out_path)

    def log_summary(self, df: pd.DataFrame) -> None:
        """Log descriptive statistics of the dataset."""
        log.info("=" * 50)
        log.info("DATASET SUMMARY")
        log.info("=" * 50)
        log.info(f"Shape          : {df.shape}")
        log.info(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        log.info(f"Null values    : {df.isnull().sum().sum()}")
        log.info(f"Duplicates     : {df.duplicated().sum()}")
        log.info(f"Dtypes:\n{df.dtypes.value_counts()}")
        log.info("=" * 50)

    def run(self) -> pd.DataFrame:
        """Full ingestion pipeline: load → validate → save → summarise."""
        log.info("Starting data ingestion pipeline...")
        df = self.load()
        self.validate_columns(df)
        self.log_summary(df)
        self.save_raw(df)
        log.info("Data ingestion complete.")
        return df


def main():
    loader = DataLoader()
    loader.run()


if __name__ == "__main__":
    main()
