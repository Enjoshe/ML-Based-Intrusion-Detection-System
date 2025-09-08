import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import numpy as np

from src.deep_model import FeedForwardNet, LSTMAutoencoder
from src.data_loader import load_data
from src.feature_engineering import preprocess_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_ffn(X_train, y_train, X_test, y_test, epochs=10, batch_size=64):
    input_dim = X_train.shape[1]
    model = FeedForwardNet(input_dim).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        print(f"[FFN] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
        y_pred = torch.argmax(preds, dim=1).cpu().numpy()
        print(classification_report(y_test, y_pred))

    return model


def train_lstm_autoencoder(X_train, seq_len=10, epochs=5, batch_size=64):
    input_dim = X_train.shape[1]
    model = LSTMAutoencoder(input_dim).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Reshape for LSTM (batch, seq_len, input_dim)
    X_seq = np.repeat(X_train[:, np.newaxis, :], seq_len, axis=1)
    dataset = TensorDataset(torch.tensor(X_seq, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, in loader:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
        print(f"[LSTM-AE] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model


if __name__ == "__main__":
    # Load + preprocess dataset
    X_train, X_test, y_train, y_test = load_data()
    X_train, X_test = preprocess_data(X_train, X_test)

    print("Training FeedForwardNet...")
    ffn_model = train_ffn(X_train, y_train, X_test, y_test)

    print("\nTraining LSTM Autoencoder...")
    ae_model = train_lstm_autoencoder(X_train)
