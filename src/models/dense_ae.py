"""
Standard and Denoising Dense (MLP) Autoencoder Baseline.
"""
from typing import Optional
import torch
import torch.nn as nn

try:
    from src.config import N_FEATURES, SEQUENCE_LENGTH, LATENT_SIZE
except ImportError:
    from config import N_FEATURES, SEQUENCE_LENGTH, LATENT_SIZE


class DenseAutoencoder(nn.Module):
    """Dense (Multilayer Perceptron) Autoencoder.

    Flattens (seq_len, n_features) and maps through fully connected layers:
      (seq_len * n_features) -> 128 -> 64 -> latent (16) -> 64 -> 128 -> (seq_len * n_features)
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        seq_len: int = SEQUENCE_LENGTH,
        latent_size: int = LATENT_SIZE,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.input_dim = seq_len * n_features
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, latent_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(128, self.input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        flat = x.view(batch_size, -1)
        return self.encoder(flat)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.size(0)
        flat_recon = self.decoder(latent)
        return flat_recon.view(batch_size, self.seq_len, self.n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction
