# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------------
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛒 Walmart Weekly Sales Forecast")
st.write("Predict weekly sales for a given store and department using a trained RandomForest pipeline.")

# --------------------------------------------------------
# Load Artifacts
# --------------------------------------------------------
MODEL_PATH = "models/rf_pipeline.pkl"
FEATURE_STATS_PATH = "models/feature_stats.json"
META_PATH = "models/meta.json"

pipeline, feature_stats, meta = None, {}, {}

try:
    pipeline = joblib.load(MODEL_PATH)
    st.sidebar.success("✅ Model pipeline loaded")
except Exception as e:
    st.sidebar.error("❌ Model pipeline not found. Please run training and add rf_pipeline.pkl")
    st.stop()

if os.path.exists(FEATURE_STATS_PATH):
    with open(FEATURE_STATS_PATH, "r") as f:
        feature_stats = json.load(f)

if os.path.exists(META_PATH):
    with open(META_PATH, "r") as f:
        meta = json.load(f)

# --------------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------------
st.sidebar.header("Input Parameters")

store = st.sidebar.number_input("Store ID", min_value=1, step=1, value=int(meta.get("stores_default", 1)))
dept = st.sidebar.number_input("Department ID", min_value=1, step=1, value=int(meta.get("depts_default", 1)))
date = st.sidebar.date_input("Week Ending Date", value=datetime.today())
is_holiday = st.sidebar.checkbox("Holiday Week?", value=False)

temp = st.sidebar.number_input("Temperature", value=60.0, format="%.2f")
fuel_price = st.sidebar.number_input("Fuel Price", value=3.5, format="%.2f")
cpi = st.sidebar.number_input("CPI", value=250.0, format="%.2f")
unemployment = st.sidebar.number_input("Unemployment Rate", value=6.0, format="%.2f")

markdown1 = st.sidebar.number_input("Markdown1", value=0.0, format="%.2f")
markdown2 = st.sidebar.number_input("Markdown2", value=0.0, format="%.2f")
markdown3 = st.sidebar.number_input("Markdown3", value=0.0, format="%.2f")
markdown4 = st.sidebar.number_input("Markdown4", value=0.0, format="%.2f")
markdown5 = st.sidebar.number_input("Markdown5", value=0.0, format="%.2f")

store_type = st.sidebar.selectbox("Store Type", options=meta.get("types", ["A", "B", "C"]))
size = st.sidebar.number_input("Store Size", value=float(meta.get("size_median", 100000)), format="%.0f")

# --------------------------------------------------------
# Feature Engineering for Input
# --------------------------------------------------------
week = int(pd.to_datetime(date).isocalendar().week)
dayofweek = int(pd.to_datetime(date).dayofweek)

# Build input row
input_row = {
    "store": int(store),
    "dept": int(dept),
    "IsHoliday": int(is_holiday),
    "temperature": float(temp),
    "fuel_price": float(fuel_price),
    "markdown1": float(markdown1),
    "markdown2": float(markdown2),
    "markdown3": float(markdown3),
    "markdown4": float(markdown4),
    "markdown5": float(markdown5),
    "cpi": float(cpi),
    "unemployment": float(unemployment),
    "type": str(store_type),
    "size": float(size),
    "week": int(week),
    "dayofweek": int(dayofweek),
    # Lag features fallback
    "weekly_sales_lagged_1": feature_stats.get("weekly_sales_lagged_1_median", 0.0),
    "weekly_sales_rolling_mean_4": feature_stats.get("weekly_sales_rolling_mean_4_median", 0.0),
    "weekly_sales_rolling_mean_12": feature_stats.get("weekly_sales_rolling_mean_12_median", 0.0),
}

X_input = pd.DataFrame([input_row])

# --------------------------------------------------------
# Prediction
# --------------------------------------------------------
if st.sidebar.button("Predict Sales"):
    try:
        y_pred = pipeline.predict(X_input)[0]
        st.subheader("📈 Prediction Result")
        st.success(f"Predicted Weekly Sales: **₹ {y_pred:,.2f}**")

        # Optional: Display input table
        st.write("### Input Features")
        st.dataframe(X_input)

    except Exception as e:
        st.error("⚠️ Prediction failed. Possible feature mismatch.")
        st.text(str(e))
        st.write("Input columns:", X_input.columns.tolist())
