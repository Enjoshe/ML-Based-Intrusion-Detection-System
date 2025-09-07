"""
model.py
Training entry point for IDS baseline.
Saves model artifact (joblib) and scaler (via feature_engineering).
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from src.data_loader import load_data
from src.feature_engineering import prepare_features

MODEL_PATH = "models/ids_model.joblib"

def train(data_path, out_dir="models", test_size=0.25, seed=42, do_cv=False):
    X, y = load_data(data_path)
    X_proc = prepare_features(X, fit_scaler=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=test_size, random_state=seed, stratify=y
    )

    if do_cv:
        param_grid = {"n_estimators": [100, 200], "max_depth": [None, 10]}
        clf = GridSearchCV(RandomForestClassifier(random_state=seed, n_jobs=-1), param_grid, cv=3, n_jobs=-1)
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Validation accuracy:", acc)
    print(classification_report(y_test, preds))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, os.path.basename(MODEL_PATH))
    joblib.dump(clf, path)
    print("Model saved to:", path)
    return path

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train IDS model")
    ap.add_argument("--data", required=True, help="CSV path with 'label' column")
    ap.add_argument("--out", default="models", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--cv", action="store_true", help="Run grid search CV (slower)")
    args = ap.parse_args()
    train(args.data, out_dir=args.out, test_size=args.test_size, seed=args.seed, do_cv=args.cv)
