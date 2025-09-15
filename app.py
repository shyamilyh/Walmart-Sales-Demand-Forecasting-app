# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="Walmart Sales Forecast", layout="wide")

# --- Helpers ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_pipeline(path="models/rf_pipeline.pkl"):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_meta(path="models/meta.json"):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_data(show_spinner=False)
def load_feature_stats(path="models/feature_stats.json"):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def compute_lag_features(df_hist, store, dept):
    """
    Attempt to compute lag1 and rolling means using historical data (df_hist).
    If not available, return None and caller should fallback to global medians.
    """
    try:
        sub = df_hist[(df_hist["store"] == int(store)) & (df_hist["dept"] == int(dept))]
        sub = sub.sort_values("date")
        if len(sub) == 0:
            return None
        last = sub.iloc[-1]
        lag1 = last["weekly_sales"]
        # rolling mean 4 and 12 weeks (if available)
        roll4 = sub["weekly_sales"].tail(4).mean() if len(sub) >= 1 else lag1
        roll12 = sub["weekly_sales"].tail(12).mean() if len(sub) >= 1 else lag1
        return {"weekly_sales_lagged_1": float(lag1),
                "weekly_sales_rolling_mean_4": float(roll4),
                "weekly_sales_rolling_mean_12": float(roll12)}
    except Exception:
        return None

# --- Load artifacts -------------------------------------------------------
st.sidebar.title("Model & Data")
pipeline = None
meta = load_meta()
feature_stats = load_feature_stats()

try:
    pipeline = load_pipeline()
except Exception as e:
    st.sidebar.error("Model pipeline not found. Run the training script (src/train_save_model.py) and put models/rf_pipeline.pkl in the repo.")
    st.sidebar.write(e)

# Try to load a sample historical CSV if available for visualizations
st.sidebar.write("Optional: place a historical dataset at `data/walmart_sales.csv` to enable lag feature computation & history plots.")
try:
    df_hist = pd.read_csv("data/walmart_sales.csv", parse_dates=["date"])
except Exception:
    df_hist = None

# --- UI -------------------------------------------------------------------
st.title("🛒 Walmart Weekly Sales Forecast")
st.markdown("Enter store, department and contextual features to predict weekly sales.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input features")
    store = st.number_input("Store (id)", min_value=1, step=1, value=int(meta.get("stores_default", 1)))
    dept = st.number_input("Department (id)", min_value=1, step=1, value=int(meta.get("depts_default", 1)))
    date = st.date_input("Forecast week end date (Friday recommended)", value=datetime.today())
    is_holiday = st.checkbox("Is Holiday week?", value=False)
    temp = st.number_input("Temperature", value=60.0, format="%.2f")
    fuel_price = st.number_input("Fuel price", value=3.5, format="%.3f")
    markdown1 = st.number_input("Markdown1", value=0.0, format="%.2f")
    markdown2 = st.number_input("Markdown2", value=0.0, format="%.2f")
    markdown3 = st.number_input("Markdown3", value=0.0, format="%.2f")
    markdown4 = st.number_input("Markdown4", value=0.0, format="%.2f")
    markdown5 = st.number_input("Markdown5", value=0.0, format="%.2f")
    cpi = st.number_input("CPI", value=250.0, format="%.2f")
    unemployment = st.number_input("Unemployment rate", value=6.0, format="%.2f")
    store_type = st.selectbox("Store type", options=meta.get("types", ["A","B","C"]), index=0)
    size = st.number_input("Store size", value=float(meta.get("size_median", 100000.0)), format="%.0f")

with col2:
    st.subheader("Prediction")
    predict_btn = st.button("Predict sales")

# Prepare input row
week = int(pd.to_datetime(date).isocalendar().week)
dayofweek = int(pd.to_datetime(date).dayofweek)

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
    "dayofweek": int(dayofweek)
}

# compute lag features using historical file if possible
lag_feats = None
if df_hist is not None:
    lag_feats = compute_lag_features(df_hist, store, dept)

# fallback to global medians saved during training
if lag_feats is None:
    lag_feats = {
        "weekly_sales_lagged_1": feature_stats.get("weekly_sales_lagged_1_median", 0.0),
        "weekly_sales_rolling_mean_4": feature_stats.get("weekly_sales_rolling_mean_4_median", 0.0),
        "weekly_sales_rolling_mean_12": feature_stats.get("weekly_sales_rolling_mean_12_median", 0.0)
    }

input_row.update(lag_feats)

# Convert to DataFrame for the pipeline
X_input = pd.DataFrame([input_row])

# Prediction
if predict_btn:
    if pipeline is None:
        st.error("No model pipeline available. See the sidebar for instructions.")
    else:
        # Predict
        y_pred = pipeline.predict(X_input)[0]
        # For RandomForest: get per-tree predictions to create a quick uncertainty interval (if model supports)
        try:
            rf = pipeline.named_steps["model"]
            preds = np.array([est.predict(X_input) for est in rf.estimators_])
            lower = float(np.percentile(preds, 5))
            upper = float(np.percentile(preds, 95))
        except Exception:
            lower = y_pred * 0.9
            upper = y_pred * 1.1

        st.metric(label="Predicted Weekly Sales", value=f"₹ {y_pred:,.2f}", delta=f"± { (upper - lower)/2:,.0f }")

        # Show quick diagnostics
        st.write("**Model uncertainty interval (approx):**", f"Lower = {lower:,.2f}, Upper = {upper:,.2f}")

        # Plot historical series for this store-dept (if available)
        if df_hist is not None:
            sub = df_hist[(df_hist["store"] == int(store)) & (df_hist["dept"] == int(dept))].sort_values("date")
            if len(sub) > 0:
                last_n = 20
                sub_plot = sub.tail(last_n)
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(sub_plot["date"], sub_plot["weekly_sales"], marker="o", label="Historical sales")
                ax.axhline(y=y_pred, color="orange", linestyle="--", label="Predicted (this request)")
                ax.set_title(f"Historical weekly sales (store {store} / dept {dept}) — last {last_n} weeks")
                ax.set_xlabel("Date")
                ax.set_ylabel("Weekly Sales")
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("No historical data found for this store & department in `data/walmart_sales.csv`.")
        else:
            st.info("Historical CSV not available (data/walmart_sales.csv). Add it to enable history plots and better lag calculation.")

st.markdown("---")
st.caption("App built from a RandomForest pipeline. For best results deploy with the trained pipeline and the original historical dataset. See README for full repo & deployment instructions.")
