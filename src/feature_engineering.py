"""
feature_engineering.py
- basic preprocessing and scaler persistence
- function: prepare_features(X_df, fit_scaler=False)
  returns processed pandas DataFrame suitable for ML training/prediction
"""

import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCALER_PATH = "models/scaler.joblib"

def add_row_stats(df):
    df = df.copy()
    df["row_mean"] = df.mean(axis=1)
    df["row_std"] = df.std(axis=1)
    df["row_max"] = df.max(axis=1)
    df["row_min"] = df.min(axis=1)
    return df

def prepare_features(X_df, fit_scaler=False):
    # Ensure numeric
    X = X_df.copy().astype(float)
    # Fill missing
    X = X.fillna(X.median())
    # Derived features
    X = add_row_stats(X)
    # Fit or load scaler
    scaler = StandardScaler()
    if fit_scaler or not os.path.exists(SCALER_PATH):
        X_scaled = scaler.fit_transform(X)
        os.makedirs(os.path.dirname(SCALER_PATH) or ".", exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)
    X_processed = pd.DataFrame(X_scaled, columns=[f"p{i}" for i in range(X_scaled.shape[1])])
    return X_processed
