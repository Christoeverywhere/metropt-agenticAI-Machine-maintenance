"""
Attention-Enhanced Denoising LSTM Autoencoder for Multivariate Time Series.

Includes:
  - Temporal Self-Attention over Encoder Hidden States
  - Bottleneck Latent Representation
  - Attention-Conditioned Decoder
  - Attention Weight Extraction for Visual Interpretability
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.config import N_FEATURES, SEQUENCE_LENGTH, LATENT_SIZE
except ImportError:
    from config import N_FEATURES, SEQUENCE_LENGTH, LATENT_SIZE


class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) Temporal Self-Attention Mechanism.

    Computes normalized attention scores over encoder hidden states:
      u_t = tanh(W_a * h_t + b_a)
      score_t = u_t * v_a
      alpha = softmax(score, dim=time)
      context = sum(alpha_t * h_t)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_states: (batch_size, seq_len, hidden_dim)
        Returns:
            context: (batch_size, hidden_dim)
            weights: (batch_size, seq_len, 1)
        """
        # (batch_size, seq_len, hidden_dim)
        u = torch.tanh(self.W_a(encoder_states))
        # (batch_size, seq_len, 1)
        scores = self.v_a(u)
        weights = F.softmax(scores, dim=1)
        # (batch_size, hidden_dim)
        context = torch.sum(weights * encoder_states, dim=1)
        return context, weights


class AttentionDenoisingLSTMAutoencoder(nn.Module):
    """Attention-Enhanced Denoising LSTM Autoencoder."""

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

        # Encoder: 2-layer LSTM returning all hidden states
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

        # Temporal Self-Attention over second LSTM hidden states
        self.attention = TemporalAttention(hidden_dim=hidden_dim2)

        # Bottleneck projection
        self.bottleneck = nn.Linear(hidden_dim2, latent_size)

        # Decoder: Latent + Context conditioning -> 2-layer Decoder LSTM
        self.expand = nn.Sequential(
            nn.Linear(latent_size + hidden_dim2, hidden_dim2),
            nn.LeakyReLU(0.1),
        )
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

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes sequence and extracts attention context.

        Returns:
            latent: (batch, latent_size)
            context: (batch, hidden_dim2)
            attn_weights: (batch, seq_len, 1)
        """
        out1, _ = self.enc_lstm1(x)
        out1 = self.enc_dropout(out1)
        out2, _ = self.enc_lstm2(out1)  # (batch, seq_len, hidden_dim2)

        context, attn_weights = self.attention(out2)
        latent = self.bottleneck(context)
        return latent, context, attn_weights

    def decode(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Decodes latent + context into full reconstruction."""
        seq_len = seq_len or self.seq_len
        batch_size = latent.size(0)

        # Combine latent representation with attention context
        combined = torch.cat([latent, context], dim=-1)  # (batch, latent_size + hidden_dim2)
        seed = self.expand(combined)                     # (batch, hidden_dim2)

        decoder_input = seed.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim2)

        dec_out1, _ = self.dec_lstm1(decoder_input)
        dec_out1 = self.dec_dropout(dec_out1)
        dec_out2, _ = self.dec_lstm2(dec_out1)
        reconstruction = self.output_proj(dec_out2)      # (batch, seq_len, n_features)
        return reconstruction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, context, _ = self.encode(x)
        reconstruction = self.decode(latent, context, seq_len=x.size(1))
        return reconstruction

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Returns attention weights (batch_size, seq_len) for visualization."""
        with torch.no_grad():
            _, _, attn_weights = self.encode(x)
        return attn_weights.squeeze(-1)
