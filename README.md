# AI-Powered Intrusion Detection System (IDS)

## Overview
This project develops an intelligent Intrusion Detection System that applies both **classical Machine Learning** and **Deep Learning (AI)** techniques to detect anomalous and malicious network behavior.  
It is designed to benchmark models across approaches and demonstrate how AI can enhance cybersecurity defenses in real-world scenarios.

## Features
- **Classical ML models:** Random Forest, Isolation Forest, Logistic Regression
- **Deep Learning models:** Feed-Forward Neural Network, LSTM Autoencoder
- **Attack categories detected:** DoS, Probe, U2R, R2L (from benchmark datasets)
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Pipeline includes:** preprocessing, feature engineering, training, evaluation, prediction

## Tech Stack
- Python 3.9+
- Scikit-learn, Pandas, NumPy
- PyTorch (Deep Learning)
- Jupyter Notebooks for experiments
- SQLite for optional log storage

## Overview
A modular, production-minded baseline ML pipeline for intrusion detection:
- Load or synthesize labeled traffic features.
- Feature processing with scaler persistence.
- Train RandomForest baseline and save artifacts.
- Evaluate with classification report, confusion matrix, and ROC/AUC.
- Predict utility for single-sample scoring.

