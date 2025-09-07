"""
synth_data.py
Generate a synthetic dataset with numeric features and binary labels for testing.
Saves CSV with columns f0..fN and a 'label' column (0=benign, 1=malicious).
"""

import argparse
import pandas as pd
from sklearn.datasets import make_classification

def generate_csv(path="data/sample_traffic.csv", n=2000, n_features=20, random_state=42):
    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=max(5, n_features // 3),
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        weights=[0.85, 0.15],
        class_sep=1.0,
        random_state=random_state
    )
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y
    df.to_csv(path, index=False)
    print(f"Wrote synthetic dataset: {path} (n={n}, features={n_features})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/sample_traffic.csv", help="CSV output path")
    parser.add_argument("--n", type=int, default=2000, help="Number of samples")
    parser.add_argument("--features", type=int, default=20, help="Number of features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    generate_csv(args.out, n=args.n, n_features=args.features, random_state=args.seed)

