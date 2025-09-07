"""
evaluate.py
Load a trained model and evaluate on dataset (CSV). Produces:
- classification_report.txt
- confusion_matrix.png
- roc_auc.png (if probabilities available)
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from src.data_loader import load_data
from src.feature_engineering import prepare_features

def evaluate(model_path, data_path, out_dir="reports"):
    model = joblib.load(model_path)
    X, y = load_data(data_path)
    X_proc = prepare_features(X, fit_scaler=False)

    preds = model.predict(X_proc)
    prob = model.predict_proba(X_proc)[:, 1] if hasattr(model, "predict_proba") else None

    # textual report
    os.makedirs(out_dir, exist_ok=True)
    report_txt = classification_report(y, preds)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report_txt)
    print(report_txt)

    # confusion matrix
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print("Saved confusion matrix to", cm_path)

    # ROC / AUC
    if prob is not None:
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0,1], [0,1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        roc_path = os.path.join(out_dir, "roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
        print("Saved ROC curve to", roc_path)
    else:
        print("Model does not support predict_proba; ROC not generated")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model .joblib")
    ap.add_argument("--data", required=True, help="CSV dataset path")
    ap.add_argument("--out", default="reports", help="Output folder")
    args = ap.parse_args()
    evaluate(args.model, args.data, out_dir=args.out)
