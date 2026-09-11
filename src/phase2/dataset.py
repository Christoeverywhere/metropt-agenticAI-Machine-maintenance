"""
Phase 2 Dataset Construction: Zero-Leakage Event-Aware Splitting,
Impending Failure Horizon Labeling (6h, 12h, 24h, 48h), and Sequence Batching.
"""
import os
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from src.config import (
        TIMESTAMP_COL,
        ANALOGUE_SENSORS,
        KNOWN_FAILURES,
        SEQUENCE_LENGTH,
        DEVICE,
    )
    from src.preprocessing import transform_data
except ImportError:
    from config import (
        TIMESTAMP_COL,
        ANALOGUE_SENSORS,
        KNOWN_FAILURES,
        SEQUENCE_LENGTH,
        DEVICE,
    )
    from preprocessing import transform_data


def create_phase2_event_splits(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the full MetroPT-3 stream into 3 strictly chronological event-aware partitions:
      - train_df: 2020-02-01 to 2020-05-15 (Normal operation + Failure #1 on 2020-04-18)
      - val_df:   2020-05-15 to 2020-06-02 (Normal operation + Failure #2 on 2020-05-29)
      - test_df:  2020-06-03 to 2020-09-01 (Normal operation + Failure #3 on Jun 5 & Failure #4 on Jul 15)

    Guarantees ZERO window or failure episode leakage across training and test.
    """
    ts = pd.to_datetime(df[TIMESTAMP_COL])

    train_cutoff = pd.Timestamp("2020-05-15 00:00:00")
    val_cutoff = pd.Timestamp("2020-06-03 00:00:00")

    train_df = df.loc[ts < train_cutoff].reset_index(drop=True)
    val_df = df.loc[(ts >= train_cutoff) & (ts < val_cutoff)].reset_index(drop=True)
    test_df = df.loc[ts >= val_cutoff].reset_index(drop=True)

    print(f"[phase2/splits] Train Split: {len(train_df):,} rows ({train_df[TIMESTAMP_COL].iloc[0]} -> {train_df[TIMESTAMP_COL].iloc[-1]}) [Failure #1]")
    print(f"[phase2/splits] Val Split:   {len(val_df):,} rows ({val_df[TIMESTAMP_COL].iloc[0]} -> {val_df[TIMESTAMP_COL].iloc[-1]}) [Failure #2]")
    print(f"[phase2/splits] Test Split:  {len(test_df):,} rows ({test_df[TIMESTAMP_COL].iloc[0]} -> {test_df[TIMESTAMP_COL].iloc[-1]}) [Failures #3 & #4]")

    return train_df, val_df, test_df


def build_phase2_supervised_windows(
    scaled_array: np.ndarray,
    df_timestamps: Union[pd.Series, np.ndarray],
    horizon_hours: float = 24.0,
    seq_len: int = SEQUENCE_LENGTH,
    stride: int = 30,
    failures: List[Dict] = KNOWN_FAILURES,
    phase1_features: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constructs supervised sequences with Impending Failure Horizon labels.

    Label Definition:
      For each window ending at reference timestamp T_ref:
        label = 1 if a documented failure begins within (0, horizon_hours] after T_ref.
        label = 0 otherwise.

    Args:
        scaled_array: (n_rows, 7) scaled sensor features
        df_timestamps: timestamps corresponding to each row
        horizon_hours: Prediction horizon in hours (e.g. 24.0)
        seq_len: Window length in timesteps (default: 180 = 30 mins)
        stride: Stride between successive sliding windows
        failures: List of known failure definitions
        phase1_features: Optional extra feature columns (e.g. Phase 1 anomaly scores)

    Returns:
        windows: (n_windows, seq_len, n_features) float32
        labels: (n_windows,) int32 (0 = Normal, 1 = Impending Failure)
        ref_timestamps: (n_windows,) datetime64 reference timestamps
    """
    n_rows, n_features = scaled_array.shape
    if phase1_features is not None:
        full_features = np.hstack([scaled_array, phase1_features]).astype(np.float32)
        n_features = full_features.shape[1]
    else:
        full_features = scaled_array

    if n_rows < seq_len:
        return np.empty((0, seq_len, n_features), dtype=np.float32), np.empty(0, dtype=np.int32), np.empty(0, dtype="datetime64[ns]")

    n_windows = (n_rows - seq_len) // stride + 1
    elem_bytes = full_features.strides[0]
    shape = (n_windows, seq_len, n_features)
    strides = (stride * elem_bytes, elem_bytes, full_features.strides[1])

    windows = np.lib.stride_tricks.as_strided(full_features, shape=shape, strides=strides)
    windows = np.ascontiguousarray(windows, dtype=np.float32)

    # Reference timestamps: last timestamp of each window
    ts_array = pd.to_datetime(df_timestamps).values
    last_indices = np.arange(n_windows) * stride + seq_len - 1
    ref_timestamps = ts_array[last_indices]

    # Generate impending failure labels
    labels = np.zeros(n_windows, dtype=np.int32)
    horizon_sec = horizon_hours * 3600.0

    for f in failures:
        f_start = np.datetime64(pd.Timestamp(f["start"]))
        f_end = np.datetime64(pd.Timestamp(f["end"]))

        # Time delta in seconds from reference time to failure start
        delta_sec = (f_start - ref_timestamps) / np.timedelta64(1, "s")

        # Positive if reference time is strictly before failure start and within horizon
        pos_mask = (delta_sec > 0) & (delta_sec <= horizon_sec)
        labels[pos_mask] = 1

    return windows, labels, ref_timestamps


class SupervisedSequenceDataset(Dataset):
    """PyTorch Dataset for Supervised Impending Failure Prediction."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).float().unsqueeze(1)  # (n_samples, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]


def create_phase2_dataloader(
    windows: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    """Creates a standard PyTorch DataLoader for supervised sequence learning."""
    dataset = SupervisedSequenceDataset(windows, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
