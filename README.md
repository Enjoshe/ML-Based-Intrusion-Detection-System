# ML-Based Intrusion Detection System (IDS)

## Overview
A modular, production-minded baseline ML pipeline for intrusion detection:
- Load or synthesize labeled traffic features.
- Feature processing with scaler persistence.
- Train RandomForest baseline and save artifacts.
- Evaluate with classification report, confusion matrix, and ROC/AUC.
- Predict utility for single-sample scoring.

## Quickstart (local)
1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
