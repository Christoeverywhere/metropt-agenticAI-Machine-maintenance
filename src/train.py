"""
Two-stage training:

  Stage 1 — Initial fit
    Train the LSTM autoencoder from scratch on the first month of
    (presumed healthy) data. This gives us a solid baseline of "normal"
    compressor behaviour before any incremental updates happen.

  Stage 2 — Incremental batch-by-batch fine-tuning
    Walk the remaining ~5 months in fixed-size chunks (default: weekly),
    simulating new data "arriving" in production. For each chunk:
      1. Score it with the current model (mean reconstruction error).
      2. If the chunk's error is NOT already anomalous (below
         FINETUNE_SKIP_ERROR_MULTIPLE x the healthy baseline), lightly
         fine-tune on it for a couple of epochs at a low learning rate.
      3. If it IS anomalous, skip fine-tuning (don't teach the model the
         fault looks normal) but keep the scored output for evaluation.
    This means the model adapts to slow seasonal/operational drift while
    refusing to absorb fault behaviour as "normal."

Run:
    python train.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
import data_loader
import windowing
from model import LSTMAutoencoder, sequence_reconstruction_error

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloader(sequences: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(sequences)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, optimizer=None) -> float:
    """One pass over `loader`. If optimizer is given, trains; else just evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_batches = 0.0, 0
    loss_fn = torch.nn.MSELoss()

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            x_hat = model(batch_x)
            loss = loss_fn(x_hat, batch_x)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def initial_fit(healthy_train_df) -> tuple[LSTMAutoencoder, object]:
    print(f"[stage 1] healthy_train rows: {len(healthy_train_df):,}")

    scaler = windowing.fit_scaler(healthy_train_df)
    scaled = windowing.scale(healthy_train_df, scaler)
    sequences = windowing.make_sequences(
        scaled, config.SEQUENCE_LENGTH, config.SEQUENCE_STRIDE
    )
    print(f"[stage 1] training sequences: {sequences.shape}")

    loader = build_dataloader(sequences, config.INITIAL_BATCH_SIZE, shuffle=True)

    model = LSTMAutoencoder(n_features=len(config.ANALOGUE_FEATURES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.INITIAL_LR)

    for epoch in range(1, config.INITIAL_EPOCHS + 1):
        loss = run_epoch(model, loader, optimizer)
        print(f"[stage 1] epoch {epoch}/{config.INITIAL_EPOCHS} — train MSE: {loss:.6f}")

    return model, scaler


def healthy_baseline_error(model, healthy_train_df, scaler) -> float:
    """Mean reconstruction error on the healthy training data itself —
    the reference point used to decide whether a new chunk looks anomalous."""
    scaled = windowing.scale(healthy_train_df, scaler)
    sequences = windowing.make_sequences(scaled, config.SEQUENCE_LENGTH, config.INFERENCE_STRIDE)
    if len(sequences) == 0:
        return 0.0

    loader = build_dataloader(sequences, config.INITIAL_BATCH_SIZE, shuffle=False)
    model.eval()
    errors = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            x_hat = model(batch_x)
            err = sequence_reconstruction_error(batch_x, x_hat)
            errors.append(err.cpu().numpy())
    return float(np.concatenate(errors).mean())


def incremental_finetune(model, scaler, stream_df, baseline_error: float):
    print(f"\n[stage 2] streaming {len(stream_df):,} rows in "
          f"'{config.FINETUNE_CHUNK}' chunks")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.FINETUNE_LR)
    skip_threshold = baseline_error * config.FINETUNE_SKIP_ERROR_MULTIPLE

    for chunk_start, chunk_end, chunk_df in data_loader.iter_stream_chunks(stream_df):
        scaled = windowing.scale(chunk_df, scaler)
        sequences = windowing.make_sequences(
            scaled, config.SEQUENCE_LENGTH, config.SEQUENCE_STRIDE
        )
        if len(sequences) == 0:
            continue

        # Score the chunk with the CURRENT model before deciding to update on it.
        loader = build_dataloader(sequences, config.FINETUNE_BATCH_SIZE, shuffle=False)
        chunk_score = run_epoch(model, loader, optimizer=None)  # eval-mode pass, no update

        label = f"{chunk_start.date()} → {chunk_end.date()}"

        if chunk_score > skip_threshold:
            print(f"[stage 2] {label}: mean err {chunk_score:.6f} > "
                  f"skip threshold {skip_threshold:.6f} — SKIPPING fine-tune "
                  f"(likely anomalous), scoring only")
            continue

        train_loader = build_dataloader(sequences, config.FINETUNE_BATCH_SIZE, shuffle=True)
        for _ in range(config.FINETUNE_EPOCHS):
            run_epoch(model, train_loader, optimizer)

        print(f"[stage 2] {label}: mean err {chunk_score:.6f} — fine-tuned "
              f"({config.FINETUNE_EPOCHS} epochs)")

    return model


def main():
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    df = data_loader.load_raw()
    healthy_train_df, stream_df = data_loader.split_healthy_train_and_stream(df)

    model, scaler = initial_fit(healthy_train_df)
    baseline_error = healthy_baseline_error(model, healthy_train_df, scaler)
    print(f"\n[baseline] healthy reconstruction error: {baseline_error:.6f}")

    model = incremental_finetune(model, scaler, stream_df, baseline_error)

    torch.save(model.state_dict(), config.MODEL_CHECKPOINT_PATH)
    with open(config.SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved model -> {config.MODEL_CHECKPOINT_PATH}")
    print(f"Saved scaler -> {config.SCALER_PATH}")


if __name__ == "__main__":
    main()
