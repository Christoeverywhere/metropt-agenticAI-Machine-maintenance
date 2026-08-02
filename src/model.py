"""
LSTM sequence autoencoder.

Encoder LSTM compresses a (seq_len, n_features) window into a latent vector.
Decoder LSTM reconstructs the full sequence from that latent vector.
Reconstruction error (per timestep, per feature) is the anomaly signal, and
per-feature error is what gives us the "which sensor is driving this" story
for the agent's explanation — no separate SHAP model needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = config.HIDDEN_SIZE,
        latent_size: int = config.LATENT_SIZE,
        num_layers: int = config.NUM_LSTM_LAYERS,
        dropout: float = config.DROPOUT,
        seq_len: int = config.SEQUENCE_LENGTH,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len

        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)

        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.encoder_lstm(x)
        last_hidden = h_n[-1]                      # (batch, hidden_size)
        latent = self.to_latent(last_hidden)        # (batch, latent_size)

        decoder_input_step = self.from_latent(latent)          # (batch, hidden_size)
        decoder_input = decoder_input_step.unsqueeze(1).repeat(1, self.seq_len, 1)

        decoded, _ = self.decoder_lstm(decoder_input)           # (batch, seq_len, hidden_size)
        reconstruction = self.output_layer(decoded)             # (batch, seq_len, n_features)
        return reconstruction


def per_feature_reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    x, x_hat: (batch, seq_len, n_features)
    returns: (batch, n_features) — mean squared error per feature, averaged
    over the sequence. This is the vector the agent reads to explain *why*
    a window was flagged (e.g. "Motor_current and Oil_temperature drove this").
    """
    return ((x - x_hat) ** 2).mean(dim=1)


def sequence_reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """(batch,) — single scalar anomaly score per sequence (mean MSE over all features/time)."""
    return ((x - x_hat) ** 2).mean(dim=(1, 2))
