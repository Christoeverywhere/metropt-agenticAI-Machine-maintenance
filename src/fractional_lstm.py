"""
Fractional LSTM Cell, Recurrent Layer, and Sequence Autoencoder.

Includes:
  - Configurable / Learnable Fractional Activation Functions
  - Custom FractionalLSTMCell with fused gate GEMM operations & proper initialization
  - Multi-layer FractionalLSTM supporting batch_first and return_sequences
  - FractionalLSTMAutoencoder designed for drop-in parity with LSTMAutoencoder
"""
import math
from typing import Callable, List, Optional, Tuple, Union

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


class FractionalActivation(nn.Module):
    """Fractional scaling activation module.

    Applies fractional order scaling alpha to the pre-activation:
      fractional_sigmoid(x) = sigmoid(alpha * x)
      fractional_tanh(x)    = tanh(alpha * x)

    Alpha can either be a fixed constant or a learnable parameter.
    """

    def __init__(self, alpha: float = 1.0, learnable: bool = False):
        super().__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha), dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))

    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.alpha * x)

    def tanh(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.alpha * x)


class FractionalLSTMCell(nn.Module):
    """Custom LSTM cell with fused weight matrices and fractional activations."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        alpha: float = 1.0,
        learnable_alpha: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.act = FractionalActivation(alpha=alpha, learnable=learnable_alpha)

        # Fused gate parameters: (input, forget, candidate/cell, output)
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Standard uniform initialization with forget gate bias = 1.0."""
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for p in self.parameters():
            if p.dim() > 0:
                nn.init.uniform_(p, -stdv, stdv)

        # Initialize forget gate bias to 1.0 for long-term memory gradient retention
        with torch.no_grad():
            self.bias_ih[self.hidden_size : 2 * self.hidden_size].fill_(1.0)
            self.bias_hh[self.hidden_size : 2 * self.hidden_size].fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single timestep.

        Args:
            x: (batch_size, input_size)
            hx: Tuple of (h_prev, c_prev), each of shape (batch_size, hidden_size)

        Returns:
            (h_next, c_next), each of shape (batch_size, hidden_size)
        """
        if hx is None:
            h_prev = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
            c_prev = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h_prev, c_prev = hx

        # Fused linear gate projection: (batch_size, 4 * hidden_size)
        gates = (x @ self.weight_ih.T + self.bias_ih) + (h_prev @ self.weight_hh.T + self.bias_hh)

        # Chunk into input, forget, cell candidate, output gates
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=-1)

        # Fractional activation
        i_t = self.act.sigmoid(i_gate)
        f_t = self.act.sigmoid(f_gate)
        g_t = self.act.tanh(g_gate)
        o_t = self.act.sigmoid(o_gate)

        # Update cell state & hidden state
        c_t = (f_t * c_prev) + (i_t * g_t)
        h_t = o_t * self.act.tanh(c_t)

        return h_t, c_t


class FractionalLSTM(nn.Module):
    """Multi-layer Fractional LSTM sequence processor."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
        return_sequences: bool = False,
        alpha: float = 1.0,
        learnable_alpha: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.return_sequences = return_sequences

        cells = [FractionalLSTMCell(input_size, hidden_size, alpha=alpha, learnable_alpha=learnable_alpha)]
        for _ in range(num_layers - 1):
            cells.append(FractionalLSTMCell(hidden_size, hidden_size, alpha=alpha, learnable_alpha=learnable_alpha))
        self.cells = nn.ModuleList(cells)

        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) if dropout > 0 and i < num_layers - 1 else nn.Identity() for i in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, input_size) if batch_first else (seq_len, batch, input)
            hx: Optional (h_0, c_0) of shape (num_layers, batch, hidden_size)

        Returns:
            output: (batch_size, seq_len, hidden_size) if return_sequences
                    else (batch_size, hidden_size)
            (h_n, c_n): (num_layers, batch_size, hidden_size)
        """
        if not self.batch_first:
            x = x.transpose(0, 1)

        batch_size, seq_len, _ = x.size()

        if hx is None:
            h_states = [torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]
            c_states = [torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]
        else:
            h_0, c_0 = hx
            h_states = [h_0[i] for i in range(self.num_layers)]
            c_states = [c_0[i] for i in range(self.num_layers)]

        step_outputs: List[torch.Tensor] = []

        for t in range(seq_len):
            current_input = x[:, t, :]
            for layer_idx in range(self.num_layers):
                cell = self.cells[layer_idx]
                h_prev = h_states[layer_idx]
                c_prev = c_states[layer_idx]

                h_next, c_next = cell(current_input, (h_prev, c_prev))
                h_states[layer_idx] = h_next
                c_states[layer_idx] = c_next

                current_input = self.dropouts[layer_idx](h_next)

            if self.return_sequences:
                step_outputs.append(current_input.unsqueeze(1))

        if self.return_sequences:
            output = torch.cat(step_outputs, dim=1)
        else:
            output = h_states[-1]

        h_n = torch.stack(h_states, dim=0)
        c_n = torch.stack(c_states, dim=0)

        return output, (h_n, c_n)


class FractionalLSTMAutoencoder(nn.Module):
    """Sequence Autoencoder utilizing Fractional LSTM layers."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        latent_size: int = LATENT_SIZE,
        num_layers: int = NUM_LSTM_LAYERS,
        dropout: float = DROPOUT,
        seq_len: int = SEQUENCE_LENGTH,
        alpha: float = 1.0,
        learnable_alpha: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.encoder_lstm = FractionalLSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            return_sequences=False,
            alpha=alpha,
            learnable_alpha=learnable_alpha,
        )
        self.bottleneck = nn.Linear(hidden_size, latent_size)
        self.expand = nn.Linear(latent_size, hidden_size)

        self.decoder_lstm = FractionalLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            return_sequences=True,
            alpha=alpha,
            learnable_alpha=learnable_alpha,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) -> latent: (batch, latent_size)"""
        last_hidden, _ = self.encoder_lstm(x)  # (batch, hidden_size)
        latent = self.bottleneck(last_hidden)  # (batch, latent_size)
        return latent

    def decode(self, latent: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """latent: (batch, latent_size) -> reconstruction: (batch, seq_len, n_features)"""
        seq_len = seq_len or self.seq_len
        expanded = self.expand(latent)  # (batch, hidden_size)
        decoder_input = expanded.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_size)

        decoder_out, _ = self.decoder_lstm(decoder_input)  # (batch, seq_len, hidden_size)
        reconstruction = self.output_layer(decoder_out)  # (batch, seq_len, n_features)
        return reconstruction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        reconstruction = self.decode(latent, seq_len=x.size(1))
        return reconstruction


if __name__ == "__main__":
    # Self-test / smoke test
    print("[FractionalLSTM] Running smoke test...")
    model = FractionalLSTMAutoencoder(alpha=0.9, learnable_alpha=True)
    dummy = torch.randn(8, SEQUENCE_LENGTH, N_FEATURES)
    out = model(dummy)
    assert out.shape == dummy.shape, f"Shape mismatch: {out.shape} vs {dummy.shape}"

    loss = ((dummy - out) ** 2).mean()
    loss.backward()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[FractionalLSTM] Forward + Backward pass OK. Output shape: {out.shape}")
    print(f"[FractionalLSTM] Total parameters: {n_params:,}")
