# src/train_save_model.py
"""
Train a RandomForest pipeline on the Walmart dataset and save artifacts:
 - models/rf_pipeline.pkl
 - models/feature_stats.json
 - models/meta.json
Make sure data/walmart_sales.csv exists (cleaned dataset).
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = "/content" # Assuming the notebook is in /content
DATA_PATH = os.path.join(ROOT, "Walmart Sales.csv") # Corrected data path
OUT_DIR = os.path.join(ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output directory: {OUT_DIR}") # Print the output directory path

def create_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['store', 'dept', 'date'])
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['dayofweek'] = df['date'].dt.dayofweek.astype(int)
    # lag and rolling (shifted) features
    df['weekly_sales_lagged_1'] = df.groupby(['store','dept'])['weekly_sales'].shift(1)
    df['weekly_sales_rolling_mean_4'] = df.groupby(['store','dept'])['weekly_sales'].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    df['weekly_sales_rolling_mean_12'] = df.groupby(['store','dept'])['weekly_sales'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    # fill markdowns
    for m in ['markdown1','markdown2','markdown3','markdown4','markdown5']:
        if m in df.columns:
            df[m] = df[m].fillna(0)
    # drop rows without lag features (first record per store/dept)
    df = df.dropna(subset=['weekly_sales_lagged_1'])
    return df

def main():
    df = pd.read_csv(DATA_PATH)
    df = create_features(df)

    features = [
        'store','dept','IsHoliday','temperature','fuel_price',
        'markdown1','markdown2','markdown3','markdown4','markdown5',
        'cpi','unemployment','type','size','week','dayofweek',
        'weekly_sales_lagged_1','weekly_sales_rolling_mean_4','weekly_sales_rolling_mean_12'
    ]
    target = 'weekly_sales'

    X = df[features]
    y = df[target]

    # split by date or random: we'll use date cutoff similar to notebook, else small random sample
    try:
        X['date'] = df['date']
        train_mask = df['date'] < '2012-04-01'
        if train_mask.sum() > 100:
            X_train = X[train_mask].drop(columns=["date"], errors="ignore") # Drop date before training
            y_train = y[train_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=["date"], errors="ignore"), y, test_size=0.2, random_state=42) # Drop date before splitting
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=["date"], errors="ignore"), y, test_size=0.2, random_state=42) # Drop date before splitting


    # define numeric and categorical features
    categorical_features = ['type', 'store', 'dept']
    numeric_features = [c for c in features if c not in categorical_features]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    model = RandomForestRegressor(n_estimators=50, max_depth = 10, n_jobs=-1, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("Training pipeline...")
    pipeline.fit(X_train, y_train) # Use X_train without date column
    print("Saving pipeline to models/rf_pipeline.pkl")
    # Check if the directory exists before saving
    if not os.path.exists(OUT_DIR):
        print(f"Error: Output directory {OUT_DIR} does not exist.")
        return # Exit the function if the directory doesn't exist
    joblib.dump(pipeline, os.path.join(OUT_DIR, "rf_pipeline.pkl"))

    # Save some feature statistics for the app to use as fallback
    feature_stats = {
        "weekly_sales_lagged_1_median": float(X_train["weekly_sales_lagged_1"].median()),
        "weekly_sales_rolling_mean_4_median": float(X_train["weekly_sales_rolling_mean_4"].median()),
        "weekly_sales_rolling_mean_12_median": float(X_train["weekly_sales_rolling_mean_12"].median()),
    }
    with open(os.path.join(OUT_DIR, "feature_stats.json"), "w") as f:
        json.dump(feature_stats, f, indent=2)

    # Save meta (unique values) for UI defaults
    meta = {
        "stores": sorted([int(s) for s in df['store'].unique()])[:200],
        "depts": sorted([int(d) for d in df['dept'].unique()])[:200],
        "types": sorted(df['type'].unique().tolist()) if 'type' in df else ["A","B","C"],  # ✅ keep strings
        "size_median": float(df['size'].median()),
        "stores_default": int(df['store'].mode()[0]),
        "depts_default": int(df['dept'].mode()[0])
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Artifacts saved under 'models/'.")

if __name__ == "__main__":
    main()
