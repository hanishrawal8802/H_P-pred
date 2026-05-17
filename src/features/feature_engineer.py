"""Feature engineering: encoding, scaling, selection, pipeline serialisation."""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer

from src.utils.logger import get_logger
from src.utils.config_loader import cfg

log = get_logger("features")


class FeatureEngineer:
    """
    Builds and fits a complete sklearn ColumnTransformer pipeline for:
      - Numeric features: imputation + StandardScaler
      - Categorical features: imputation + OneHotEncoder
      - Feature selection: SelectKBest (mutual_info_classif)
    Serialises the fitted pipeline for inference.
    """

    def __init__(self):
        self.processed_path = cfg.get("data.processed_data_path", "data/processed/telco_churn_processed.csv")
        self.target_column = cfg.get("data.target_column", "Churn Label")
        self.pipeline_path = cfg.get("features.pipeline_path", "models/feature_pipeline.pkl")
        self.k_features: int = cfg.get("features.feature_selection.k_features") or 20
        self.feature_selection_enabled: bool = cfg.get("features.feature_selection.enabled", True)

        # Columns to exclude from features
        self._exclude = {self.target_column}

    def load_data(self) -> pd.DataFrame:
        path = Path(self.processed_path)
        if not path.exists():
            raise FileNotFoundError(f"Processed data not found: {path}")
        return pd.read_csv(path)

    def identify_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Auto-detect numerical and categorical columns."""
        feature_df = df.drop(columns=list(self._exclude), errors="ignore")
        num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
        log.info(f"Numerical features: {len(num_cols)}, Categorical features: {len(cat_cols)}")
        return num_cols, cat_cols

    def build_pipeline(self, num_cols: List[str], cat_cols: List[str]) -> Pipeline:
        """Construct the full sklearn transformation pipeline."""
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols),
            ],
            remainder="drop",
        )

        steps = [("preprocessor", preprocessor)]

        if self.feature_selection_enabled:
            steps.append((
                "selector",
                SelectKBest(score_func=mutual_info_classif, k=self.k_features),
            ))

        pipeline = Pipeline(steps=steps)
        return pipeline

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Fit the pipeline and return transformed features + labels."""
        X = df.drop(columns=list(self._exclude), errors="ignore")
        y = df[self.target_column].values

        num_cols, cat_cols = self.identify_columns(df)
        self.pipeline = self.build_pipeline(num_cols, cat_cols)

        log.info("Fitting feature engineering pipeline...")
        X_transformed = self.pipeline.fit_transform(X, y)
        log.info(f"Transformed shape: {X_transformed.shape}")

        return X_transformed, y, num_cols, cat_cols

    def get_feature_names(self, num_cols: List[str], cat_cols: List[str]) -> List[str]:
        """Extract feature names after one-hot encoding + selection."""
        try:
            preprocessor = self.pipeline.named_steps["preprocessor"]
            cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
            all_names = num_cols + cat_names

            if self.feature_selection_enabled:
                selector = self.pipeline.named_steps["selector"]
                selected_mask = selector.get_support()
                feature_names = [n for n, s in zip(all_names, selected_mask) if s]
            else:
                feature_names = all_names

            log.info(f"Selected {len(feature_names)} features.")
            return feature_names
        except Exception as exc:
            log.warning(f"Could not extract feature names: {exc}")
            return []

    def save_pipeline(self) -> str:
        """Serialise the fitted pipeline to disk."""
        out_path = Path(self.pipeline_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, out_path)
        log.info(f"Pipeline saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
        return str(out_path)

    @staticmethod
    def load_pipeline(pipeline_path: str) -> Pipeline:
        """Load a serialised pipeline from disk."""
        return joblib.load(pipeline_path)

    def save_reference_data(self, df: pd.DataFrame) -> None:
        """Save a sample of processed data as reference for drift detection."""
        ref_path = Path(cfg.get("monitoring.reference_data_path", "data/processed/reference_data.csv"))
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        sample = df.sample(min(1000, len(df)), random_state=42)
        sample.to_csv(ref_path, index=False)
        log.info(f"Reference data saved: {ref_path} ({len(sample)} rows)")

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        log.info("Starting feature engineering pipeline...")
        df = self.load_data()
        X, y, num_cols, cat_cols = self.fit_transform(df)
        self.get_feature_names(num_cols, cat_cols)
        self.save_pipeline()
        self.save_reference_data(df)
        log.info("Feature engineering complete.")
        return X, y


def main():
    fe = FeatureEngineer()
    fe.run()


if __name__ == "__main__":
    main()
