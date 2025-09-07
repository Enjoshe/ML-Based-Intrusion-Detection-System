"""
data_loader.py
- load_data(path=None)
  If CSV exists at path, load it (expects numeric features + 'label' column).
  If path is None or missing, raises FileNotFoundError (synth_data.py can create a dataset).
- returns: X (pandas DataFrame of features), y (1D numpy array)
"""

import os
import pandas as pd
import numpy as np

def load_from_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column with 0=benign,1=malicious")
    y = df["label"].values
    X = df.drop(columns=["label"]).reset_index(drop=True)
    # Ensure features are numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, y

def load_data(path=None):
    """
    Returns: X (DataFrame), y (np array)
    """
    if path:
        return load_from_csv(path)
    else:
        raise FileNotFoundError("No dataset path provided. Run synth_data.py to generate a sample CSV.")
