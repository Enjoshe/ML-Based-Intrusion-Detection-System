"""
predict.py
Load model and predict on a comma-separated sample vector supplied on CLI.
Usage:
python src/predict.py --model models/ids_model.joblib --sample "0.1,0.2,0.3,..." 
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from src.feature_engineering import prepare_features

def parse_sample(sample_str):
    parts = [float(x.strip()) for x in sample_str.split(",") if x.strip() != ""]
    return np.array(parts).reshape(1, -1)

def predict_sample(model_path, sample_str):
    model = joblib.load(model_path)
    sample = parse_sample(sample_str)
    # Put into DataFrame with numeric columns; feature_engineering will try to scale
    df = pd.DataFrame(sample, columns=[f"f{i}" for i in range(sample.shape[1])])
    try:
        Xp = prepare_features(df, fit_scaler=False)
    except Exception:
        # fallback to raw
        Xp = pd.DataFrame(sample)
    pred = model.predict(Xp)
    proba = model.predict_proba(Xp)[:, 1] if hasattr(model, "predict_proba") else None
    return int(pred[0]), float(proba[0]) if proba is not None else None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model file (.joblib)")
    ap.add_argument("--sample", required=True, help="Comma-separated numeric feature vector")
    args = ap.parse_args()
    label, score = predict_sample(args.model, args.sample)
    print("Prediction (1=malicious,0=benign):", label)
    if score is not None:
        print("Probability:", score)
