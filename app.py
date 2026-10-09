
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Himalayan Guardian",
    page_icon="🏔️",
    layout="wide"
)

DATA = Path(__file__).parent / "data"

st.title("🏔️ Himalayan Guardian")
st.subheader("Glacial Lake Outburst Flood (GLOF) Risk Monitoring")
st.write("Monitor glacial lake risk scores, locations, and environmental forecasts.")

def load_csv(filename):
    path = DATA / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

lakes = load_csv("lake_risk.csv")
villages = load_csv("villages.csv")
forecast = load_csv("forecast.csv")

summary_path = DATA / "summary.json"
summary = {}
if summary_path.exists():
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

if lakes.empty:
    st.error("No lake data found. Check data/lake_risk.csv.")
    st.stop()

high = int((lakes["risk_level"] == "High").sum()) if "risk_level" in lakes else 0
medium = int((lakes["risk_level"] == "Medium").sum()) if "risk_level" in lakes else 0

c1, c2, c3 = st.columns(3)
c1.metric("Lakes analysed", len(lakes))
c2.metric("High-risk lakes", high)
c3.metric("Medium-risk lakes", medium)

st.divider()
st.header("Lake Risk Assessment")

if "risk_level" in lakes.columns:
    options = ["All"] + sorted(lakes["risk_level"].dropna().unique().tolist())
    selected = st.selectbox("Filter by risk level", options)
    shown = lakes if selected == "All" else lakes[lakes["risk_level"] == selected]
else:
    shown = lakes

if {"lat", "lon"}.issubset(shown.columns):
    points = shown.dropna(subset=["lat", "lon"])
    if not points.empty:
        fig = px.scatter_geo(
            points,
            lat="lat",
            lon="lon",
            color="risk_level" if "risk_level" in points.columns else None,
            hover_name="name" if "name" in points.columns else None,
            scope="asia",
            title="Glacial Lake Locations"
        )
        st.plotly_chart(fig, use_container_width=True)

st.dataframe(shown, use_container_width=True)

if "risk_level" in lakes.columns:
    counts = lakes["risk_level"].value_counts().rename_axis("Risk Level").reset_index(name="Lake Count")
    st.subheader("Risk Distribution")
    st.plotly_chart(
        px.bar(counts, x="Risk Level", y="Lake Count"),
        use_container_width=True
    )

st.divider()
st.header("Environmental Forecasts")

if not forecast.empty and {"date", "variable", "value"}.issubset(forecast.columns):
    variables = forecast["variable"].dropna().unique().tolist()
    if variables:
        variable = st.selectbox("Choose forecast", variables)
        chart = forecast[forecast["variable"] == variable].copy()
        chart["date"] = pd.to_datetime(chart["date"], errors="coerce")
        fig = px.line(
            chart.sort_values("date"),
            x="date",
            y="value",
            color="kind" if "kind" in chart.columns else None,
            title=variable
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Forecast data is not available.")

st.divider()
st.header("Village Information")
if not villages.empty:
    st.dataframe(villages, use_container_width=True)
else:
    st.info("No village data available.")


st.header("Model Performance")

model_info = summary.get("model", {})
cv = model_info.get("cv", {})

c1, c2, c3 = st.columns(3)
c1.metric("Selected Model", model_info.get("name", "N/A").replace("_", " ").title())
c2.metric("ROC-AUC", f"{cv.get('cv_roc_auc', 0):.3f}")
c3.metric("Precision", f"{cv.get('cv_precision', 0):.3f}")

st.subheader("Model Comparison")

all_models = model_info.get("all_models", {})
if all_models:
    comparison = pd.DataFrame(all_models).T
    comparison.index.name = "Model"
    comparison = comparison.reset_index()
    comparison["Model"] = comparison["Model"].str.replace("_", " ").str.title()

    if "cv_roc_auc" in comparison.columns:
        fig = px.bar(
            comparison,
            x="Model",
            y="cv_roc_auc",
            title="ROC-AUC Comparison",
            range_y=[0, 1],
            text_auto=".3f"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(comparison, use_container_width=True)

st.caption(
    "Evaluation metrics are based on spatial cross-validation. "
    "They do not represent calibrated probabilities of an actual flood."
)

st.caption(
    "Research prototype: risk scores are relative model outputs, "
    "not calibrated probabilities of an actual flood."
)
