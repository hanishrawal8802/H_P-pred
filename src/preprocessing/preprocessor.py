"""Data preprocessing: cleaning, type casting, target encoding."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from src.utils.logger import get_logger
from src.utils.config_loader import cfg

log = get_logger("preprocessing")


class DataPreprocessor:
    """
    Cleans raw data and prepares it for feature engineering.
    Steps:
        1. Remove unwanted columns
        2. Fix data types (especially 'Total Charges')
        3. Encode target variable (Yes → 1, No → 0)
        4. Handle missing values
        5. Remove duplicates
        6. Save processed CSV
    """

    def __init__(self):
        self.raw_path = cfg.get("data.raw_data_path", "data/raw/telco_churn_raw.csv")
        self.processed_path = cfg.get("data.processed_data_path", "data/processed/telco_churn_processed.csv")
        self.target_column = cfg.get("data.target_column", "Churn Label")
        self.drop_columns: List[str] = cfg.get("data.drop_columns") or []
        self.categorical_columns: List[str] = cfg.get("data.categorical_columns") or []
        self.numerical_columns: List[str] = cfg.get("data.numerical_columns") or []

    def load(self) -> pd.DataFrame:
        path = Path(self.raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Raw data not found: {path}. Run ingestion first.")
        df = pd.read_csv(path)
        log.info(f"Loaded raw data: {df.shape}")
        return df

    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that are IDs or irrelevant to prediction."""
        to_drop = [c for c in self.drop_columns if c in df.columns]
        # Also drop columns that are near-constant (>99% same value)
        for col in df.columns:
            if col in to_drop or col == self.target_column:
                continue
            if df[col].nunique() <= 1:
                to_drop.append(col)
                log.warning(f"Dropping near-constant column: '{col}'")
        df = df.drop(columns=to_drop, errors="ignore")
        log.info(f"Dropped {len(to_drop)} columns. Remaining: {df.shape[1]}")
        return df

    def fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix known type issues in the IBM Telco dataset."""
        # 'Total Charges' is sometimes stored as string with spaces
        for col in ["Total Charges", "Total Long Distance Charges", "Total Revenue"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                log.debug(f"Converted '{col}' to numeric. Nulls introduced: {df[col].isnull().sum()}")

        # 'Senior Citizen' may be 'Yes'/'No' or 1/0
        if "Senior Citizen" in df.columns:
            if df["Senior Citizen"].dtype == object:
                df["Senior Citizen"] = df["Senior Citizen"].map({"Yes": 1, "No": 0})

        return df

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode target column: Yes → 1, No → 0."""
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found.")
        df[self.target_column] = df[self.target_column].map({"Yes": 1, "No": 0})
        if df[self.target_column].isnull().any():
            log.warning("Nulls in target after encoding. Dropping those rows.")
            df = df.dropna(subset=[self.target_column])
        df[self.target_column] = df[self.target_column].astype(int)
        dist = df[self.target_column].value_counts(normalize=True) * 100
        log.info(f"Target distribution (%):\n{dist.round(2)}")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with appropriate strategies."""
        for col in df.columns:
            n_null = df[col].isnull().sum()
            if n_null == 0:
                continue
            if col in self.numerical_columns or df[col].dtype in [np.float64, np.int64, float, int]:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                log.debug(f"Filled '{col}' nulls with median={median_val:.4f}")
            else:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
                log.debug(f"Filled '{col}' nulls with mode='{mode_val}'")
        log.info(f"Missing value handling complete. Total nulls: {df.isnull().sum().sum()}")
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed > 0:
            log.warning(f"Removed {removed} duplicate rows.")
        return df

    def save(self, df: pd.DataFrame) -> str:
        out = Path(self.processed_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        log.info(f"Processed data saved: {out} ({len(df):,} rows × {len(df.columns)} cols)")
        return str(out)

    def run(self) -> pd.DataFrame:
        log.info("Starting preprocessing pipeline...")
        df = self.load()
        df = self.drop_unnecessary_columns(df)
        df = self.fix_data_types(df)
        df = self.encode_target(df)
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        self.save(df)
        log.info("Preprocessing complete.")
        return df


def main():
    preprocessor = DataPreprocessor()
    preprocessor.run()


if __name__ == "__main__":
    main()
