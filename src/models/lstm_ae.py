"""
Hierarchical 2-Layer LSTM Autoencoder for Multivariate Time Series.

Architecture:
  Input (batch, seq_len, n_features)
    -> Encoder LSTM 1 (hidden_size=64, return_sequences=True)
    -> Dropout
    -> Encoder LSTM 2 (hidden_size=32, return_sequences=False)
    -> Linear Bottleneck (32 -> 16) [latent representation]
    -> Linear Expand (16 -> 32)
    -> Repeat across all seq_len timesteps
    -> Decoder LSTM 1 (hidden_size=32, return_sequences=True)
    -> Dropout
    -> Decoder LSTM 2 (hidden_size=64, return_sequences=True)
    -> Linear Output Projection (64 -> n_features)
    -> Reconstruction (batch, seq_len, n_features)
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn

try:
    from src.config import N_FEATURES, SEQUENCE_LENGTH, LATENT_SIZE
except ImportError:
    from config import N_FEATURES, SEQUENCE_LENGTH, LATENT_SIZE


class LSTMAutoencoder(nn.Module):
    """Hierarchical LSTM Sequence Autoencoder."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        seq_len: int = SEQUENCE_LENGTH,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        latent_size: int = LATENT_SIZE,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.latent_size = latent_size

        # Encoder: 2 stacked LSTM layers
        self.enc_lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim1,
            batch_first=True,
        )
        self.enc_dropout = nn.Dropout(dropout)
        self.enc_lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            batch_first=True,
        )
        self.bottleneck = nn.Linear(hidden_dim2, latent_size)

        # Decoder: Latent expand + 2 stacked LSTM layers
        self.expand = nn.Linear(latent_size, hidden_dim2)
        self.dec_lstm1 = nn.LSTM(
            input_size=hidden_dim2,
            hidden_size=hidden_dim2,
            batch_first=True,
        )
        self.dec_dropout = nn.Dropout(dropout)
        self.dec_lstm2 = nn.LSTM(
            input_size=hidden_dim2,
            hidden_size=hidden_dim1,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim1, n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) -> latent: (batch, latent_size)"""
        out1, _ = self.enc_lstm1(x)
        out1 = self.enc_dropout(out1)
        _, (h_n2, _) = self.enc_lstm2(out1)
        last_hidden = h_n2[-1]  # (batch, hidden_dim2)
        latent = self.bottleneck(last_hidden)  # (batch, latent_size)
        return latent

    def decode(self, latent: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """latent: (batch, latent_size) -> reconstruction: (batch, seq_len, n_features)"""
        seq_len = seq_len or self.seq_len
        batch_size = latent.size(0)

        expanded = self.expand(latent)  # (batch, hidden_dim2)
        # Repeat across all timesteps to feed the decoder
        decoder_input = expanded.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim2)

        dec_out1, _ = self.dec_lstm1(decoder_input)
        dec_out1 = self.dec_dropout(dec_out1)
        dec_out2, _ = self.dec_lstm2(dec_out1)
        reconstruction = self.output_proj(dec_out2)  # (batch, seq_len, n_features)
        return reconstruction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        reconstruction = self.decode(latent, seq_len=x.size(1))
        return reconstruction
