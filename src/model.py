"""
LSTM sequence autoencoder.

Architecture:
  (batch, 180, 7)
    -> Encoder LSTM (hidden_size=64)
    -> last hidden state (batch, 64)
    -> Linear bottleneck (64 -> 16)   [latent]
    -> Linear expand (16 -> 64), repeated across all 180 timesteps
    -> Decoder LSTM (hidden_size=64)
    -> Linear output (64 -> 7)
    -> reconstruction (batch, 180, 7)
"""
import torch
import torch.nn as nn

try:
    from src.config import (
        N_FEATURES,
        HIDDEN_SIZE,
        LATENT_SIZE,
        NUM_LSTM_LAYERS,
        DROPOUT,
        SEQUENCE_LENGTH,
    )
except ImportError:
    from config import (
        N_FEATURES,
        HIDDEN_SIZE,
        LATENT_SIZE,
        NUM_LSTM_LAYERS,
        DROPOUT,
        SEQUENCE_LENGTH,
    )

class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        latent_size: int = LATENT_SIZE,
        num_layers: int = NUM_LSTM_LAYERS,
        dropout: float = DROPOUT,
        seq_len: int = SEQUENCE_LENGTH,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        # LSTM dropout only applies between layers; with num_layers=1 it's a no-op,
        # so guard against the (harmless but noisy) PyTorch warning.
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.bottleneck = nn.Linear(hidden_size, latent_size)
        self.expand = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) -> latent: (batch, latent_size)"""
        _, (h_n, _) = self.encoder_lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size) - final layer's hidden state
        latent = self.bottleneck(last_hidden)  # (batch, latent_size)
        return latent

    def decode(self, latent: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """latent: (batch, latent_size) -> reconstruction: (batch, seq_len, n_features)"""
        seq_len = seq_len or self.seq_len
        batch_size = latent.size(0)

        expanded = self.expand(latent)  # (batch, hidden_size)
        # repeat across every timestep to feed the decoder
        decoder_input = expanded.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_size)

        decoder_out, _ = self.decoder_lstm(decoder_input)  # (batch, seq_len, hidden_size)
        reconstruction = self.output_layer(decoder_out)  # (batch, seq_len, n_features)
        return reconstruction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        reconstruction = self.decode(latent, seq_len=x.size(1))
        return reconstruction


def per_feature_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Mean squared error per feature, averaged over the time dimension.

    x, x_hat: (batch, seq_len, n_features)
    returns:  (batch, n_features)
    """
    return ((x - x_hat) ** 2).mean(dim=1)


def sequence_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Scalar reconstruction error per sequence (mean over time AND features).

    x, x_hat: (batch, seq_len, n_features)
    returns:  (batch,)
    """
    return ((x - x_hat) ** 2).mean(dim=(1, 2))


if __name__ == "__main__":
    # Quick smoke test with random data - shape checks only.
    model = LSTMAutoencoder()
    dummy = torch.randn(8, SEQUENCE_LENGTH, N_FEATURES)
    recon = model(dummy)
    assert recon.shape == dummy.shape, f"Shape mismatch: {recon.shape} vs {dummy.shape}"

    pf_err = per_feature_error(dummy, recon)
    seq_err = sequence_error(dummy, recon)
    assert pf_err.shape == (8, N_FEATURES)
    assert seq_err.shape == (8,)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Forward pass OK. Output shape: {recon.shape}")
    print(f"[model] per_feature_error shape: {pf_err.shape}, sequence_error shape: {seq_err.shape}")
    print(f"[model] Total trainable parameters: {n_params:,}")
