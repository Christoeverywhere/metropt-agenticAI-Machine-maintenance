"""
Scaling + sliding-window sequence construction.

- StandardScaler is fit ONLY on healthy_train's analogue sensor columns.
- Windows are built with numpy's stride_tricks (vectorized) - NOT a Python
  loop - because a naive loop over ~1.3M stream rows is unacceptably slow.
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from src.config import (
        ANALOGUE_SENSORS,
        SEQUENCE_LENGTH,
        SEQUENCE_STRIDE,
        INFERENCE_STRIDE,
        SCALER_PATH,
        _TORCH_AVAILABLE,
    )
except ImportError:
    from config import (
        ANALOGUE_SENSORS,
        SEQUENCE_LENGTH,
        SEQUENCE_STRIDE,
        INFERENCE_STRIDE,
        SCALER_PATH,
        _TORCH_AVAILABLE,
    )

if _TORCH_AVAILABLE:
    import torch


def fit_scaler(healthy_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on healthy_train's analogue sensor columns only."""
    scaler = StandardScaler()
    scaler.fit(healthy_train[ANALOGUE_SENSORS].values)
    return scaler


def save_scaler(scaler: StandardScaler, path: str = SCALER_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str = SCALER_PATH) -> StandardScaler:
    with open(path, "rb") as f:
        return pickle.load(f)


def transform(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Apply a fitted scaler to a DataFrame's analogue sensor columns.
    Returns a (n_rows, n_features) float32 array."""
    return scaler.transform(df[ANALOGUE_SENSORS].values).astype(np.float32)


def build_windows(scaled_array: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    """Vectorized sliding-window construction.

    scaled_array: (n_rows, n_features) float32
    returns:      (n_sequences, seq_len, n_features) float32

    Uses numpy.lib.stride_tricks.sliding_window_view, which creates a VIEW
    (no data copy) over the windows dimension, then slices by stride and
    copies only the final (much smaller) result. This is dramatically
    faster than a Python-level loop over rows for ~1M+ row arrays.
    """
    n_rows, n_features = scaled_array.shape
    if n_rows < seq_len:
        raise ValueError(
            f"Cannot build windows of length {seq_len} from only {n_rows} rows."
        )

    # sliding_window_view over the time axis -> shape (n_rows - seq_len + 1, n_features, seq_len)
    windows = np.lib.stride_tricks.sliding_window_view(scaled_array, seq_len, axis=0)
    # -> (n_windows, n_features, seq_len); transpose last two dims to (n_windows, seq_len, n_features)
    windows = windows.transpose(0, 2, 1)

    if stride > 1:
        windows = windows[::stride]

    # Ensure contiguous + float32 (sliding_window_view can leave non-contiguous strided views)
    return np.ascontiguousarray(windows, dtype=np.float32)


def build_training_windows(healthy_train: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Windows for Stage 1 training: stride = SEQUENCE_STRIDE."""
    scaled = transform(healthy_train, scaler)
    return build_windows(scaled, SEQUENCE_LENGTH, SEQUENCE_STRIDE)


def build_inference_windows(df: pd.DataFrame, scaler: StandardScaler):
    """Dense windows for scoring/inference: stride = INFERENCE_STRIDE.

    Also returns the timestamps corresponding to each window's LAST
    timestep, since evaluate.py attributes each sequence's error to its
    final timestep (real-time scoring semantics)."""
    scaled = transform(df, scaler)
    windows = build_windows(scaled, SEQUENCE_LENGTH, INFERENCE_STRIDE)

    timestamps = df["timestamp"].values
    # window i covers rows [i*stride : i*stride+seq_len); its "last timestep"
    # timestamp is at index (i*stride + seq_len - 1)
    n_windows = windows.shape[0]
    last_idx = np.arange(n_windows) * INFERENCE_STRIDE + SEQUENCE_LENGTH - 1
    window_timestamps = timestamps[last_idx]

    return windows, window_timestamps


def to_tensor(windows: np.ndarray):
    """Convert a numpy window array to a torch.FloatTensor. Requires PyTorch."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed in this environment. to_tensor() requires "
            "torch - install PyTorch to use this function (see requirements.txt)."
        )
    return torch.from_numpy(windows).float()


if __name__ == "__main__":
    try:
        from src.data_loader import load_and_split
    except ImportError:
        from data_loader import load_and_split

    healthy_train, stream = load_and_split()

    scaler = fit_scaler(healthy_train)
    save_scaler(scaler)
    print(f"[windowing] Fitted StandardScaler on {len(healthy_train):,} healthy rows, "
          f"saved to {SCALER_PATH}")
    print(f"[windowing] Scaler means: {dict(zip(ANALOGUE_SENSORS, scaler.mean_.round(3)))}")

    train_windows = build_training_windows(healthy_train, scaler)
    print(f"[windowing] Training windows (stride={SEQUENCE_STRIDE}): {train_windows.shape}")

    # Sanity: build dense inference windows on a SMALL slice (first 2000 rows)
    # to prove the inference path works without doing it over the full stream
    # here (that's evaluate.py's job, and it's slow without a GPU).
    small_slice = stream.iloc[:2000].reset_index(drop=True)
    infer_windows, infer_ts = build_inference_windows(small_slice, scaler)
    print(f"[windowing] Inference windows on 2,000-row sample (stride={INFERENCE_STRIDE}): "
          f"{infer_windows.shape}, timestamps range {infer_ts[0]} -> {infer_ts[-1]}")

    assert not np.isnan(train_windows).any(), "NaNs found in training windows!"
    assert not np.isnan(infer_windows).any(), "NaNs found in inference windows!"
    print("[windowing] Sanity checks passed: no NaNs in either window set.")
