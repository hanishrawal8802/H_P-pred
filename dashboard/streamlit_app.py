"""Streamlit dashboard for the Telecom Churn MLOps Platform."""
import os
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn MLOps Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://api:8000")
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
    border: 1px solid #3a3a5e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.metric-value { font-size: 2.2rem; font-weight: 700; color: #a78bfa; }
.metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }

.churn-badge-yes {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white; padding: 8px 20px; border-radius: 20px;
    font-weight: 700; font-size: 1.1rem; display: inline-block;
}
.churn-badge-no {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white; padding: 8px 20px; border-radius: 20px;
    font-weight: 700; font-size: 1.1rem; display: inline-block;
}

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #1e1e2e; border-radius: 8px 8px 0 0;
    border: 1px solid #3a3a5e; color: #94a3b8; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def check_api_health() -> dict:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json()
    except Exception:
        return {"status": "unreachable", "model_loaded": False}


def make_prediction(features: dict) -> dict:
    try:
        r = requests.post(f"{API_URL}/predict", json=features, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def load_metrics_from_file() -> dict:
    path = Path("models/training_metadata.json")
    if path.exists():
        return json.loads(path.read_text())
    return {}


def load_drift_summary() -> dict:
    reports = sorted(Path("monitoring/evidently/reports").glob("*.json"), reverse=True)
    if reports:
        return json.loads(reports[0].read_text())
    return {}


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 MLOps Platform")
    st.markdown("---")
    health = check_api_health()
    status_color = "🟢" if health.get("model_loaded") else "🔴"
    st.markdown(f"{status_color} **API Status:** {health.get('status', 'unknown').upper()}")
    st.markdown(f"🔖 **Model Version:** {health.get('model_version', 'N/A')}")
    st.markdown(f"🏷️ **Stage:** {health.get('model_stage', 'N/A')}")
    st.markdown("---")
    st.markdown(f"🔗 [MLflow UI]({MLFLOW_URL})")
    st.markdown(f"🔗 [API Docs]({API_URL}/docs)")
    st.markdown(f"📊 [Grafana](http://grafana:3000)")


# ─────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Churn Predictor",
    "📊 Model Metrics",
    "🌊 Drift Monitor",
    "📈 Analytics",
])


# ─────────────────────────────────────────
# TAB 1: Churn Predictor
# ─────────────────────────────────────────
with tab1:
    st.markdown("## 🎯 Customer Churn Predictor")
    st.markdown("Fill in the customer details to get a churn prediction.")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        with col2:
            st.markdown("**Services**")
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

        with col3:
            st.markdown("**Contract & Billing**")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        st.markdown("**Financials**")
        fc1, fc2, fc3, fc4 = st.columns(4)
        tenure = fc1.slider("Tenure (months)", 0, 72, 12)
        monthly = fc2.number_input("Monthly Charges ($)", 0.0, 200.0, 75.0, step=0.5)
        total_charges = fc3.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly), step=10.0)
        cltv = fc4.number_input("CLTV", 0, 10000, 3500, step=100)

        submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True)

    if submitted:
        features = {
            "gender": gender,
            "senior_citizen": senior,
            "partner": partner,
            "dependents": dependents,
            "internet_service": internet_service,
            "online_security": online_security,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "contract": contract,
            "payment_method": payment_method,
            "paperless_billing": paperless,
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total_charges,
            "cltv": cltv,
        }

        with st.spinner("Running prediction..."):
            result = make_prediction(features)

        if "error" in result:
            st.error(f"Prediction failed: {result['error']}")
        else:
            st.markdown("---")
            r1, r2, r3 = st.columns(3)
            prediction = result.get("prediction", "Unknown")
            probability = result.get("probability", 0.5)
            confidence = result.get("confidence", "Medium")

            badge_class = "churn-badge-yes" if prediction == "Churn" else "churn-badge-no"
            r1.markdown(
                f"**Prediction**<br><span class='{badge_class}'>{prediction}</span>",
                unsafe_allow_html=True,
            )
            r2.metric("Churn Probability", f"{probability:.1%}")
            r3.metric("Confidence", confidence)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Churn Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef4444" if probability >= 0.5 else "#10b981"},
                    "steps": [
                        {"range": [0, 40], "color": "#d1fae5"},
                        {"range": [40, 70], "color": "#fef3c7"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": 50},
                },
            ))
            fig.update_layout(height=300, margin=dict(t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────
# TAB 2: Model Metrics
# ─────────────────────────────────────────
with tab2:
    st.markdown("## 📊 Model Performance Metrics")
    meta = load_metrics_from_file()

    if meta:
        metrics = meta.get("metrics", {})
        all_results = meta.get("all_results", {})

        m1, m2, m3, m4, m5 = st.columns(5)
        for col, (key, label) in zip(
            [m1, m2, m3, m4, m5],
            [("accuracy","Accuracy"),("precision","Precision"),("recall","Recall"),("f1_score","F1-Score"),("roc_auc","ROC-AUC")],
        ):
            val = metrics.get(key, 0)
            col.metric(label, f"{val:.4f}")

        st.markdown("---")
        st.markdown(f"**Best Model:** `{meta.get('model_name', 'N/A')}`  |  **Run ID:** `{meta.get('run_id', 'N/A')}`")

        if all_results:
            st.markdown("### Model Comparison")
            comparison_df = pd.DataFrame(all_results).T.reset_index()
            comparison_df.columns = ["Model"] + list(comparison_df.columns[1:])
            fig = px.bar(
                comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                x="Metric", y="Score", color="Model", barmode="group",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig.update_layout(yaxis_range=[0, 1], template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No training metadata found. Run `make train` to train models first.")


# ─────────────────────────────────────────
# TAB 3: Drift Monitor
# ─────────────────────────────────────────
with tab3:
    st.markdown("## 🌊 Data Drift Monitor")
    drift = load_drift_summary()

    if drift:
        d1, d2, d3 = st.columns(3)
        d1.metric("Dataset Drift", "⚠️ YES" if drift.get("dataset_drift") else "✅ NO")
        d2.metric("Drifted Features", drift.get("n_drifted_features", 0))
        d3.metric("Share Drifted", f"{drift.get('share_drifted_features', 0):.1%}")

        scores = drift.get("feature_drift_scores", {})
        if scores:
            df_scores = pd.DataFrame(
                {"Feature": list(scores.keys()), "Drift Score": list(scores.values())}
            ).sort_values("Drift Score", ascending=False)
            fig = px.bar(
                df_scores, x="Drift Score", y="Feature", orientation="h",
                color="Drift Score", color_continuous_scale="RdYlGn_r",
            )
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

        # Embed HTML report if available
        report_path = drift.get("report_path")
        if report_path and Path(report_path).exists():
            st.markdown("### Full Evidently Report")
            with open(report_path, "r") as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)
    else:
        st.info("No drift report found. Run `make drift` to generate a drift report.")


# ─────────────────────────────────────────
# TAB 4: Analytics
# ─────────────────────────────────────────
with tab4:
    st.markdown("## 📈 Prediction Analytics")

    try:
        from sqlalchemy import create_engine, text
        db_url = (
            f"postgresql+psycopg2://"
            f"{os.getenv('POSTGRES_USER','mlops')}:{os.getenv('POSTGRES_PASSWORD','mlops_secret')}"
            f"@{os.getenv('POSTGRES_HOST','postgres')}:{os.getenv('POSTGRES_PORT','5432')}"
            f"/{os.getenv('POSTGRES_DB','mlops_db')}"
        )
        engine = create_engine(db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            df_preds = pd.read_sql("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 500", conn)

        if not df_preds.empty:
            a1, a2, a3 = st.columns(3)
            a1.metric("Total Predictions", len(df_preds))
            churn_rate = (df_preds["prediction"] == "Churn").mean()
            a2.metric("Churn Rate", f"{churn_rate:.1%}")
            a3.metric("Avg Probability", f"{df_preds['probability'].mean():.3f}")

            # Prediction distribution
            fig1 = px.histogram(
                df_preds, x="probability", nbins=20,
                color_discrete_sequence=["#7c3aed"],
                title="Prediction Probability Distribution",
            )
            fig1.update_layout(template="plotly_dark")
            st.plotly_chart(fig1, use_container_width=True)

            # Timeline
            df_preds["created_at"] = pd.to_datetime(df_preds["created_at"])
            df_time = df_preds.groupby(df_preds["created_at"].dt.date)["prediction"].count().reset_index()
            df_time.columns = ["Date", "Count"]
            fig2 = px.line(df_time, x="Date", y="Count", title="Predictions Over Time")
            fig2.update_layout(template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No predictions logged yet. Use the Predictor tab to make predictions.")
    except Exception as exc:
        st.warning(f"Database not available ({exc}). Analytics requires PostgreSQL.")
