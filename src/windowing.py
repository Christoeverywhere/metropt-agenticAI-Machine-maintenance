"""
Turn a flat dataframe of sensor readings into overlapping fixed-length
sequences for the LSTM autoencoder, plus the scaler fit/apply helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


def fit_scaler(healthy_train_df: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on healthy data only — this defines 'normal'."""
    scaler = StandardScaler()
    scaler.fit(healthy_train_df[config.ANALOGUE_FEATURES].values)
    return scaler


def scale(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(df[config.ANALOGUE_FEATURES].values).astype("float32")


def make_sequences(values: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    """
    values: (n_timesteps, n_features) scaled array
    returns: (n_sequences, seq_len, n_features)
    """
    n = values.shape[0]
    if n < seq_len:
        return np.empty((0, seq_len, values.shape[1]), dtype="float32")

    starts = range(0, n - seq_len + 1, stride)
    sequences = np.stack([values[s : s + seq_len] for s in starts])
    return sequences.astype("float32")


def sequence_start_indices(n_timesteps: int, seq_len: int, stride: int) -> np.ndarray:
    """Row indices (into the original, unscaled df) that each sequence starts at —
    used to map a sequence's reconstruction error back to real timestamps."""
    if n_timesteps < seq_len:
        return np.array([], dtype=int)
    return np.arange(0, n_timesteps - seq_len + 1, stride)
