import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        enc_out, (h, c) = self.encoder(x)
        z = self.latent(enc_out[:, -1, :]).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(z)
        recon = self.output_layer(dec_out)
        return recon


class FeedForwardNet(nn.Module):
    """Simple fully connected classifier for baseline"""
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super(FeedForwardNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)
