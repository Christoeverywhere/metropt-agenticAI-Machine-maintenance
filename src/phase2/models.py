"""
Phase 2 Supervised Predictive Model Architectures:
  1. Logistic Regression Baseline
  2. 2-Layer LSTM Failure Predictor
  3. 2-Layer GRU Failure Predictor
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.config import N_FEATURES, SEQUENCE_LENGTH
except ImportError:
    from config import N_FEATURES, SEQUENCE_LENGTH


class LogisticRegressionBaseline(nn.Module):
    """Linear / Logistic Regression Baseline for Sequence Anomaly / Failure Prediction.
    Extracts summary statistics (mean, std, min, max, diff) across time and maps to a single logit.
    """

    def __init__(self, input_dim: int = N_FEATURES, seq_len: int = SEQUENCE_LENGTH):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        # Summary statistics per channel: mean (1), std (1), min (1), max (1), trend (1) -> 5 * input_dim
        self.feature_dim = 5 * input_dim
        self.linear = nn.Linear(self.feature_dim, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) -> (batch, 5 * input_dim)"""
        mean_feat = x.mean(dim=1)
        std_feat = x.std(dim=1, unbiased=False)
        min_feat, _ = x.min(dim=1)
        max_feat, _ = x.max(dim=1)
        # Trend: difference between second half and first half of the sequence
        half = self.seq_len // 2
        trend_feat = x[:, half:, :].mean(dim=1) - x[:, :half, :].mean(dim=1)

        feats = torch.cat([mean_feat, std_feat, min_feat, max_feat, trend_feat], dim=-1)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (batch, 1)."""
        feats = self.extract_features(x)
        return self.linear(feats)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns failure probabilities (batch, 1) in [0, 1]."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


class LSTMFailurePredictor(nn.Module):
    """Supervised 2-Layer LSTM Failure Classifier.

    Architecture:
      Input (batch, seq_len, input_dim)
        -> LSTM 1 (hidden_size=64, return_sequences=True)
        -> Dropout(0.2)
        -> LSTM 2 (hidden_size=32, return_sequences=False)
        -> Dropout(0.2)
        -> Linear(32, 16) -> ReLU
        -> Linear(16, 1)  -> Logits
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        dense_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            batch_first=True,
        )
        self.drop2 = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim2, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (batch, 1)."""
        out1, _ = self.lstm1(x)
        out1 = self.drop1(out1)
        _, (h_n2, _) = self.lstm2(out1)
        last_hidden = h_n2[-1]  # (batch, hidden_dim2)
        last_hidden = self.drop2(last_hidden)
        logits = self.head(last_hidden)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns failure probabilities (batch, 1) in [0, 1]."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


class GRUFailurePredictor(nn.Module):
    """Supervised 2-Layer GRU Failure Classifier (Controlled Parameter Parity with LSTM).

    Architecture:
      Input (batch, seq_len, input_dim)
        -> GRU 1 (hidden_size=64, return_sequences=True)
        -> Dropout(0.2)
        -> GRU 2 (hidden_size=32, return_sequences=False)
        -> Dropout(0.2)
        -> Linear(32, 16) -> ReLU
        -> Linear(16, 1)  -> Logits
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        dense_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.gru1 = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            batch_first=True,
        )
        self.drop2 = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim2, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (batch, 1)."""
        out1, _ = self.gru1(x)
        out1 = self.drop1(out1)
        _, h_n2 = self.gru2(out1)
        last_hidden = h_n2[-1]  # (batch, hidden_dim2)
        last_hidden = self.drop2(last_hidden)
        logits = self.head(last_hidden)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns failure probabilities (batch, 1) in [0, 1]."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
