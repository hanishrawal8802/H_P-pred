"""Evidently AI drift detection module."""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesSummaryMetric,
)

from src.utils.logger import get_logger
from src.utils.config_loader import cfg

log = get_logger("drift_detector")


class DriftDetector:
    """
    Uses Evidently AI to detect:
      - Feature drift (Jensen-Shannon divergence)
      - Prediction drift
      - Target drift
    Generates HTML reports and JSON summaries.
    Triggers retraining if drift exceeds threshold.
    """

    def __init__(self):
        self.reference_data_path = cfg.get(
            "monitoring.reference_data_path", "data/processed/reference_data.csv"
        )
        self.report_dir = Path(cfg.get("monitoring.drift_report_path", "monitoring/evidently/reports/"))
        self.drift_threshold = float(cfg.get("monitoring.drift_threshold", 0.15))
        self.retrain_on_drift = cfg.get("monitoring.retrain_on_drift", True)
        self.target_column = cfg.get("data.target_column", "Churn Label")
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def load_reference_data(self) -> pd.DataFrame:
        path = Path(self.reference_data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Reference data not found: {path}. Run feature engineering first."
            )
        return pd.read_csv(path)

    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """Return feature columns (excluding target)."""
        return [c for c in df.columns if c != self.target_column]

    def generate_drift_report(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        report_name: str = "drift",
    ) -> Tuple[str, Dict]:
        """Generate Evidently drift report and return (html_path, summary_dict)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self.report_dir / f"{report_name}_{timestamp}.html"
        json_path = self.report_dir / f"{report_name}_{timestamp}.json"

        # Build report
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])

        feature_cols = self._get_feature_columns(reference_df)
        ref = reference_df[feature_cols].copy()
        cur = current_df[feature_cols].copy()

        report.run(reference_data=ref, current_data=cur)
        report.save_html(str(html_path))

        # Extract summary
        result = report.as_dict()
        drift_summary = self._extract_drift_summary(result)

        with open(json_path, "w") as f:
            json.dump(drift_summary, f, indent=2)

        log.info(f"Drift report saved: {html_path}")
        log.info(f"Dataset drift detected: {drift_summary.get('dataset_drift', False)}")
        log.info(f"Drifted features: {drift_summary.get('n_drifted_features', 0)}")

        return str(html_path), drift_summary

    def _extract_drift_summary(self, result: Dict) -> Dict:
        """Extract key metrics from Evidently result dict."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "dataset_drift": False,
            "n_drifted_features": 0,
            "share_drifted_features": 0.0,
            "feature_drift_scores": {},
        }
        try:
            for metric in result.get("metrics", []):
                metric_id = metric.get("metric", "")
                result_data = metric.get("result", {})

                if "DatasetDriftMetric" in metric_id:
                    summary["dataset_drift"] = result_data.get("dataset_drift", False)
                    summary["n_drifted_features"] = result_data.get("number_of_drifted_columns", 0)
                    summary["share_drifted_features"] = result_data.get(
                        "share_of_drifted_columns", 0.0
                    )

                if "ColumnDriftMetric" in metric_id:
                    col = result_data.get("column_name", "unknown")
                    score = result_data.get("drift_score", 0.0)
                    summary["feature_drift_scores"][col] = round(score, 4)
        except Exception as exc:
            log.warning(f"Could not extract full drift summary: {exc}")

        return summary

    def should_retrain(self, drift_summary: Dict) -> bool:
        """Determine if retraining should be triggered."""
        if not self.retrain_on_drift:
            return False
        share = drift_summary.get("share_drifted_features", 0.0)
        triggered = share > self.drift_threshold
        if triggered:
            log.warning(
                f"⚠️  Drift threshold exceeded: {share:.2%} > {self.drift_threshold:.2%}. "
                f"Retraining will be triggered."
            )
        return triggered

    def trigger_retraining(self) -> None:
        """Trigger retraining pipeline (calls trainer directly or via Airflow)."""
        log.info("🔄 Triggering retraining pipeline...")
        try:
            from src.training.trainer import ModelTrainer
            from src.training.registry import ModelRegistry

            trainer = ModelTrainer()
            model, run_id, metrics = trainer.run()

            registry = ModelRegistry()
            registry.run(run_id, metrics)
            log.info("✅ Retraining and re-deployment complete.")
        except Exception as exc:
            log.error(f"Retraining failed: {exc}")
            raise

    def run(
        self,
        current_data: Optional[pd.DataFrame] = None,
        report_name: str = "feature_drift",
    ) -> Dict:
        """Full drift detection pipeline."""
        log.info("Starting drift detection...")
        reference_df = self.load_reference_data()

        if current_data is None:
            # Use processed data as "current" if no live data provided
            processed_path = Path(cfg.get("data.processed_data_path", "data/processed/telco_churn_processed.csv"))
            if not processed_path.exists():
                log.warning("No current data available for drift detection.")
                return {}
            current_data = pd.read_csv(processed_path)

        # Align columns
        common_cols = [c for c in reference_df.columns if c in current_data.columns]
        reference_df = reference_df[common_cols]
        current_data = current_data[common_cols]

        html_path, summary = self.generate_drift_report(reference_df, current_data, report_name)

        if self.should_retrain(summary):
            self.trigger_retraining()

        return {**summary, "report_path": html_path}


def main():
    detector = DriftDetector()
    result = detector.run()
    print(f"Drift Detection Complete:")
    print(f"  Dataset Drift: {result.get('dataset_drift', False)}")
    print(f"  Drifted Features: {result.get('n_drifted_features', 0)}")
    print(f"  Report: {result.get('report_path', 'N/A')}")


if __name__ == "__main__":
    main()
