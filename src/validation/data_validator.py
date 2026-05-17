"""Data validation using Great Expectations with manual fallback."""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.config_loader import cfg

log = get_logger("validation")


class DataValidator:
    """
    Validates the raw dataset against a defined schema.
    Uses custom validation rules (GE-compatible design) for portability.
    """

    SCHEMA: Dict[str, Any] = {
        "CustomerID": {"dtype": "object", "nullable": False, "unique": True},
        "Gender": {"dtype": "object", "nullable": True, "allowed": ["Male", "Female"]},
        "Senior Citizen": {"dtype": "object", "nullable": True},
        "Tenure Months": {"dtype": "numeric", "nullable": True, "min": 0, "max": 120},
        "Monthly Charges": {"dtype": "numeric", "nullable": True, "min": 0},
        "Total Charges": {"dtype": "numeric", "nullable": True, "min": 0},
        "Churn Label": {"dtype": "object", "nullable": False, "allowed": ["Yes", "No"]},
    }

    def __init__(self):
        self.raw_data_path = cfg.get("data.raw_data_path", "data/raw/telco_churn_raw.csv")
        self.validation_report_path = cfg.get("data.validation_report_path", "data/validation/")
        self.target_column = cfg.get("data.target_column", "Churn Label")
        self.results: List[Dict] = []

    def load_data(self) -> pd.DataFrame:
        path = Path(self.raw_data_path)
        if not path.exists():
            raise FileNotFoundError(f"Raw data not found at {path}. Run ingestion first.")
        return pd.read_csv(path)

    # ─────────────────────────────────────────
    # Individual checks
    # ─────────────────────────────────────────
    def _check_not_empty(self, df: pd.DataFrame) -> Dict:
        passed = len(df) > 0
        return {
            "check": "Dataset not empty",
            "passed": passed,
            "details": f"Row count: {len(df):,}",
        }

    def _check_no_duplicates(self, df: pd.DataFrame) -> Dict:
        n_dup = df.duplicated().sum()
        return {
            "check": "No duplicate rows",
            "passed": n_dup == 0,
            "details": f"Duplicates found: {n_dup}",
        }

    def _check_target_exists(self, df: pd.DataFrame) -> Dict:
        passed = self.target_column in df.columns
        return {
            "check": f"Target column '{self.target_column}' exists",
            "passed": passed,
            "details": f"Columns: {list(df.columns)[:5]}...",
        }

    def _check_target_values(self, df: pd.DataFrame) -> Dict:
        if self.target_column not in df.columns:
            return {"check": "Target values valid", "passed": False, "details": "Column missing"}
        vals = set(df[self.target_column].dropna().unique())
        expected = {"Yes", "No"}
        passed = vals.issubset(expected)
        return {
            "check": "Target values are Yes/No",
            "passed": passed,
            "details": f"Found: {vals}",
        }

    def _check_missing_values(self, df: pd.DataFrame) -> List[Dict]:
        results = []
        for col in df.columns:
            null_pct = df[col].isnull().mean() * 100
            threshold = 80.0
            passed = null_pct < threshold
            results.append({
                "check": f"Missing values < {threshold}% in '{col}'",
                "passed": passed,
                "details": f"{null_pct:.2f}% missing",
            })
        return results

    def _check_schema(self, df: pd.DataFrame) -> List[Dict]:
        results = []
        for col, rules in self.SCHEMA.items():
            if col not in df.columns:
                results.append({
                    "check": f"Schema: column '{col}' exists",
                    "passed": False,
                    "details": "Column not found in dataset",
                })
                continue
            # Nullable
            if not rules.get("nullable", True):
                n_null = df[col].isnull().sum()
                results.append({
                    "check": f"Schema: '{col}' not nullable",
                    "passed": n_null == 0,
                    "details": f"Null count: {n_null}",
                })
            # Allowed values
            if "allowed" in rules:
                vals = set(df[col].dropna().unique())
                invalid = vals - set(rules["allowed"])
                results.append({
                    "check": f"Schema: '{col}' has valid categories",
                    "passed": len(invalid) == 0,
                    "details": f"Invalid values: {invalid}" if invalid else "OK",
                })
            # Numeric range
            if rules.get("dtype") == "numeric":
                try:
                    s = pd.to_numeric(df[col], errors="coerce")
                    if "min" in rules:
                        passed = (s.dropna() >= rules["min"]).all()
                        results.append({
                            "check": f"Schema: '{col}' >= {rules['min']}",
                            "passed": bool(passed),
                            "details": f"Min value: {s.min():.2f}",
                        })
                except Exception:
                    pass
        return results

    def _check_unique(self, df: pd.DataFrame) -> Dict:
        id_col = cfg.get("data.customer_id_column", "CustomerID")
        if id_col not in df.columns:
            return {"check": f"'{id_col}' unique", "passed": True, "details": "Column not found"}
        n_unique = df[id_col].nunique()
        passed = n_unique == len(df)
        return {
            "check": f"'{id_col}' is unique",
            "passed": passed,
            "details": f"Unique: {n_unique}, Total: {len(df)}",
        }

    # ─────────────────────────────────────────
    # Report generation
    # ─────────────────────────────────────────
    def generate_html_report(self, results: List[Dict]) -> str:
        passed = sum(1 for r in results if r["passed"])
        failed = len(results) - passed
        report_dir = Path(self.validation_report_path)
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = report_dir / f"validation_report_{timestamp}.html"

        rows = ""
        for r in results:
            status = "✅ PASS" if r["passed"] else "❌ FAIL"
            color = "#d4edda" if r["passed"] else "#f8d7da"
            rows += (
                f"<tr style='background:{color}'>"
                f"<td>{r['check']}</td><td>{status}</td><td>{r.get('details','')}</td></tr>"
            )

        html = f"""<!DOCTYPE html>
<html><head><title>Data Validation Report</title>
<style>body{{font-family:Arial;margin:40px}} table{{width:100%;border-collapse:collapse}}
th,td{{border:1px solid #ddd;padding:10px;text-align:left}}
th{{background:#343a40;color:white}} h1{{color:#343a40}}</style></head>
<body>
<h1>📊 Data Validation Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<p>✅ Passed: <b>{passed}</b> &nbsp; ❌ Failed: <b>{failed}</b> &nbsp; Total: <b>{len(results)}</b></p>
<table><tr><th>Check</th><th>Status</th><th>Details</th></tr>{rows}</table>
</body></html>"""

        html_path.write_text(html)
        log.info(f"Validation report saved: {html_path}")
        return str(html_path)

    def run(self) -> Tuple[bool, str]:
        """Execute all validation checks and return (passed, report_path)."""
        log.info("Starting data validation...")
        df = self.load_data()

        results = []
        results.append(self._check_not_empty(df))
        results.append(self._check_no_duplicates(df))
        results.append(self._check_target_exists(df))
        results.append(self._check_target_values(df))
        results.append(self._check_unique(df))
        results.extend(self._check_missing_values(df))
        results.extend(self._check_schema(df))

        self.results = results
        passed = all(r["passed"] for r in results)
        failed_checks = [r for r in results if not r["passed"]]

        report_path = self.generate_html_report(results)

        total = len(results)
        n_pass = sum(1 for r in results if r["passed"])
        log.info(f"Validation complete: {n_pass}/{total} checks passed.")
        if failed_checks:
            log.warning(f"Failed checks: {[r['check'] for r in failed_checks]}")

        return passed, report_path


def main():
    validator = DataValidator()
    passed, report = validator.run()
    print(f"Validation {'PASSED' if passed else 'FAILED'} — Report: {report}")


if __name__ == "__main__":
    main()
