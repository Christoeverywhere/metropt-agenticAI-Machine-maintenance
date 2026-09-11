"""
Sliding-Window Sequence Construction and PyTorch Denoising Dataset.
"""
from typing import Optional, Tuple
import numpy as np
from numpy.lib.stride_tricks import as_strided
import torch
from torch.utils.data import Dataset, DataLoader


def build_sliding_windows(
    scaled_array: np.ndarray,
    seq_len: int = 180,
    stride: int = 30,
) -> np.ndarray:
    """Efficiently builds 3D sliding windows (n_windows, seq_len, n_features)
    using numpy stride tricks (zero-copy until materialized)."""
    n_rows, n_features = scaled_array.shape
    if n_rows < seq_len:
        return np.empty((0, seq_len, n_features), dtype=np.float32)

    n_windows = (n_rows - seq_len) // stride + 1
    elem_bytes = scaled_array.strides[0]

    shape = (n_windows, seq_len, n_features)
    strides = (stride * elem_bytes, elem_bytes, scaled_array.strides[1])

    # Contiguous copy to ensure independent memory layout
    windows = as_strided(scaled_array, shape=shape, strides=strides)
    return np.ascontiguousarray(windows, dtype=np.float32)


class DenoisingWindowDataset(Dataset):
    """PyTorch Dataset for Autoencoder training.

    Supports:
      - Clean reconstruction: inputs = targets = x
      - Denoising reconstruction: inputs = noisy(x), targets = x
    """

    def __init__(
        self,
        windows: np.ndarray,
        noise_std: float = 0.0,
        mask_prob: float = 0.0,
        is_training: bool = True,
    ):
        self.targets = torch.from_numpy(windows).float()
        self.noise_std = float(noise_std)
        self.mask_prob = float(mask_prob)
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean_seq = self.targets[idx]  # (seq_len, n_features)

        if self.is_training and (self.noise_std > 0.0 or self.mask_prob > 0.0):
            noisy_seq = clean_seq.clone()

            # 1. Add controlled Gaussian jitter
            if self.noise_std > 0.0:
                noise = torch.randn_like(noisy_seq) * self.noise_std
                noisy_seq = noisy_seq + noise

            # 2. Add random timestep masking (temporal interpolation challenge)
            if self.mask_prob > 0.0:
                mask = torch.rand(noisy_seq.size(0), 1) < self.mask_prob
                noisy_seq = noisy_seq.masked_fill(mask, 0.0)

            return noisy_seq, clean_seq
        else:
            return clean_seq, clean_seq


def create_dataloader(
    windows: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = True,
    noise_std: float = 0.0,
    mask_prob: float = 0.0,
    is_training: bool = True,
) -> DataLoader:
    """Helper to instantiate a standard DataLoader with DenoisingWindowDataset."""
    dataset = DenoisingWindowDataset(
        windows=windows,
        noise_std=noise_std,
        mask_prob=mask_prob,
        is_training=is_training,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
