"""
Two-stage training.

Stage 1 (initial_fit):       train from scratch on healthy_train (month 1).
Stage 2 (incremental_finetune): walk the stream in 7-day chunks; score each
    chunk with the current model, then EITHER skip it (if it looks anomalous,
    error > 2.5x healthy baseline) OR fine-tune on it for 2 epochs at a lower
    learning rate (to absorb legitimate drift without catastrophic forgetting
    and without teaching the model that faults are normal).

Both functions accept an optional `progress_callback(event: dict)` so a
caller (e.g. a future backend service) can get live progress instead of only
parsed stdout. Running this file directly (`python src/train.py`) reproduces
exactly the console output documented in the README.
"""
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.config import (
        INITIAL_EPOCHS, INITIAL_LR, INITIAL_BATCH_SIZE,
        FINETUNE_CHUNK, FINETUNE_EPOCHS, FINETUNE_LR, FINETUNE_BATCH_SIZE,
        FINETUNE_SKIP_ERROR_MULTIPLE,
        SEQUENCE_LENGTH, SEQUENCE_STRIDE,
        MODEL_PATH, SCALER_PATH, RANDOM_SEED, DEVICE,
    )
    from src.data_loader import load_and_split
    from src.windowing import fit_scaler, save_scaler, transform, build_windows, to_tensor
    from src.model import LSTMAutoencoder, sequence_error
except ImportError:
    from config import (
        INITIAL_EPOCHS, INITIAL_LR, INITIAL_BATCH_SIZE,
        FINETUNE_CHUNK, FINETUNE_EPOCHS, FINETUNE_LR, FINETUNE_BATCH_SIZE,
        FINETUNE_SKIP_ERROR_MULTIPLE,
        SEQUENCE_LENGTH, SEQUENCE_STRIDE,
        MODEL_PATH, SCALER_PATH, RANDOM_SEED, DEVICE,
    )
    from data_loader import load_and_split
    from windowing import fit_scaler, save_scaler, transform, build_windows, to_tensor
    from model import LSTMAutoencoder, sequence_error


def _set_seed(seed: int = RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _noop_callback(event: dict):
    pass


def _run_epochs(model, windows: np.ndarray, epochs: int, lr: float, batch_size: int,
                 label: str, progress_callback=None):
    """Shared training loop used by both stages. Returns list of per-epoch mean MSE."""
    progress_callback = progress_callback or _noop_callback

    tensor = to_tensor(windows).to(DEVICE)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    epoch_losses = []
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, n_batches = 0.0, 0
        for (batch_x,) in loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = loss_fn(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        mean_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(mean_loss)
        print(f"[{label}] epoch {epoch}/{epochs} — train MSE: {mean_loss:.6f}")
        progress_callback({
            "stage": label, "epoch": epoch, "epochs": epochs, "train_mse": mean_loss,
        })

    return epoch_losses


@torch.no_grad()
def _score_windows(model, windows: np.ndarray, batch_size: int = 512) -> float:
    """Mean sequence reconstruction error over a set of windows, no gradient."""
    model.eval()
    tensor = to_tensor(windows).to(DEVICE)
    errors = []
    for i in range(0, len(tensor), batch_size):
        batch = tensor[i:i + batch_size]
        recon = model(batch)
        err = sequence_error(batch, recon)
        errors.append(err.cpu().numpy())
    return float(np.concatenate(errors).mean())


def initial_fit(healthy_train: pd.DataFrame, progress_callback=None):
    """Stage 1: train from scratch on healthy_train. Returns (model, scaler, baseline_error)."""
    _set_seed()
    progress_callback = progress_callback or _noop_callback

    scaler = fit_scaler(healthy_train)
    save_scaler(scaler)

    scaled = transform(healthy_train, scaler)
    windows = build_windows(scaled, SEQUENCE_LENGTH, SEQUENCE_STRIDE)

    print(f"[stage 1] healthy_train rows: {len(healthy_train):,}")
    print(f"[stage 1] training sequences: {windows.shape}")
    progress_callback({"stage": "stage1_start", "n_rows": len(healthy_train), "window_shape": windows.shape})

    model = LSTMAutoencoder().to(DEVICE)
    _run_epochs(model, windows, INITIAL_EPOCHS, INITIAL_LR, INITIAL_BATCH_SIZE,
                label="stage 1", progress_callback=progress_callback)

    baseline_error = _score_windows(model, windows)
    print(f"[baseline] healthy reconstruction error: {baseline_error:.6f}")
    progress_callback({"stage": "baseline", "baseline_error": baseline_error})

    return model, scaler, baseline_error


def _iter_chunks(stream: pd.DataFrame, freq: str = FINETUNE_CHUNK):
    """Yield (chunk_start, chunk_end, chunk_df) walking the stream chronologically
    in `freq`-sized windows (e.g. '7D')."""
    start = stream["timestamp"].iloc[0].floor("D")
    end = stream["timestamp"].iloc[-1]
    boundaries = pd.date_range(start=start, end=end + pd.Timedelta(freq), freq=freq)

    for i in range(len(boundaries) - 1):
        chunk_start, chunk_end = boundaries[i], boundaries[i + 1]
        mask = (stream["timestamp"] >= chunk_start) & (stream["timestamp"] < chunk_end)
        chunk_df = stream.loc[mask]
        if len(chunk_df) == 0:
            continue
        yield chunk_start, chunk_end, chunk_df.reset_index(drop=True)


def incremental_finetune(model, scaler, stream: pd.DataFrame, baseline_error: float,
                          progress_callback=None):
    """Stage 2: walk the stream in chunks, score + selectively fine-tune."""
    progress_callback = progress_callback or _noop_callback
    print(f"[stage 2] streaming {len(stream):,} rows in '{FINETUNE_CHUNK}' chunks")
    progress_callback({"stage": "stage2_start", "n_rows": len(stream)})

    chunk_log = []
    skip_threshold = baseline_error * FINETUNE_SKIP_ERROR_MULTIPLE

    for chunk_start, chunk_end, chunk_df in _iter_chunks(stream):
        if len(chunk_df) < SEQUENCE_LENGTH:
            print(f"[stage 2] {chunk_start.date()} -> {chunk_end.date()}: "
                  f"skipped (only {len(chunk_df)} rows, need >= {SEQUENCE_LENGTH})")
            continue

        scaled = transform(chunk_df, scaler)
        windows = build_windows(scaled, SEQUENCE_LENGTH, SEQUENCE_STRIDE)

        chunk_error = _score_windows(model, windows)
        will_finetune = chunk_error <= skip_threshold

        if will_finetune:
            _run_epochs(model, windows, FINETUNE_EPOCHS, FINETUNE_LR, FINETUNE_BATCH_SIZE,
                        label=f"stage 2 | {chunk_start.date()}", progress_callback=progress_callback)
            action = f"fine-tuned ({FINETUNE_EPOCHS} epochs)"
        else:
            action = f"SKIPPED (error > {FINETUNE_SKIP_ERROR_MULTIPLE}x baseline)"

        print(f"[stage 2] {chunk_start.date()} -> {chunk_end.date()}: "
              f"mean err {chunk_error:.6f} — {action}")

        chunk_log.append({
            "chunk_start": chunk_start, "chunk_end": chunk_end,
            "n_rows": len(chunk_df), "mean_error": chunk_error,
            "fine_tuned": will_finetune,
        })
        progress_callback({
            "stage": "stage2_chunk", "chunk_start": str(chunk_start), "chunk_end": str(chunk_end),
            "mean_error": chunk_error, "fine_tuned": will_finetune,
        })

    return model, chunk_log


def save_checkpoint(model, baseline_error: float, chunk_log: list, path: str = MODEL_PATH):
    torch.save({
        "model_state_dict": model.state_dict(),
        "baseline_error": baseline_error,
        "chunk_log": chunk_log,
        "model_config": {
            "n_features": model.n_features,
            "hidden_size": model.hidden_size,
            "latent_size": model.latent_size,
            "num_layers": model.num_layers,
            "seq_len": model.seq_len,
        },
    }, path)
    print(f"[train] Saved checkpoint to {path}")


def main(progress_callback=None):
    t0 = time.time()
    healthy_train, stream = load_and_split()

    model, scaler, baseline_error = initial_fit(healthy_train, progress_callback=progress_callback)
    model, chunk_log = incremental_finetune(model, scaler, stream, baseline_error,
                                             progress_callback=progress_callback)

    save_checkpoint(model, baseline_error, chunk_log)
    save_scaler(scaler)

    elapsed = time.time() - t0
    n_finetuned = sum(1 for c in chunk_log if c["fine_tuned"])
    print(f"\n[train] Done in {elapsed/60:.1f} min. "
          f"{n_finetuned}/{len(chunk_log)} chunks fine-tuned, "
          f"{len(chunk_log) - n_finetuned} skipped as anomalous.")

    return model, scaler, baseline_error, chunk_log


if __name__ == "__main__":
    main()
