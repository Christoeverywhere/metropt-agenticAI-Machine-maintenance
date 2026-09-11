"""
Side-by-Side Benchmark: Standard PyTorch LSTM Autoencoder vs. Fractional LSTM Autoencoder.

Compares:
  1. Trainable Parameter Count
  2. Training Loss Convergence (MSE)
  3. Training & Inference Throughput (windows/sec, ms/batch)
  4. Healthy Baseline Reconstruction MSE (mean +/- std)
  5. Anomaly Detection Lead Time on Known MetroPT Failures
"""
import argparse
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.config import (
        N_FEATURES,
        HIDDEN_SIZE,
        LATENT_SIZE,
        NUM_LSTM_LAYERS,
        DROPOUT,
        SEQUENCE_LENGTH,
        SEQUENCE_STRIDE,
        INITIAL_EPOCHS,
        INITIAL_LR,
        INITIAL_BATCH_SIZE,
        THRESHOLD_K,
        KNOWN_FAILURES,
        RANDOM_SEED,
        DEVICE,
    )
    from src.data_loader import load_and_split
    from src.windowing import fit_scaler, transform, build_windows, to_tensor
    from src.model import LSTMAutoencoder, sequence_error
    from src.fractional_lstm import FractionalLSTMAutoencoder
    from src.evaluate import calibrate_threshold, evaluate_lead_times
except ImportError:
    from config import (
        N_FEATURES,
        HIDDEN_SIZE,
        LATENT_SIZE,
        NUM_LSTM_LAYERS,
        DROPOUT,
        SEQUENCE_LENGTH,
        SEQUENCE_STRIDE,
        INITIAL_EPOCHS,
        INITIAL_LR,
        INITIAL_BATCH_SIZE,
        THRESHOLD_K,
        KNOWN_FAILURES,
        RANDOM_SEED,
        DEVICE,
    )
    from data_loader import load_and_split
    from windowing import fit_scaler, transform, build_windows, to_tensor
    from model import LSTMAutoencoder, sequence_error
    from fractional_lstm import FractionalLSTMAutoencoder
    from evaluate import calibrate_threshold, evaluate_lead_times


def set_seed(seed: int = RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_profile_model(
    model: nn.Module,
    model_name: str,
    train_windows: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Dict[str, Any]:
    """Trains a model while recording timing and loss metrics."""
    set_seed(RANDOM_SEED)
    model = model.to(DEVICE)
    tensor = to_tensor(train_windows).to(DEVICE)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    epoch_losses: List[float] = []
    epoch_durations: List[float] = []
    total_samples = len(train_windows) * epochs

    print(f"\n--- Training {model_name} ({epochs} epochs, {len(train_windows):,} windows) ---")
    start_total = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss, n_batches = 0.0, 0

        for (batch_x,) in loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = loss_fn(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        epoch_duration = time.perf_counter() - epoch_start
        epoch_durations.append(epoch_duration)
        mean_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(mean_loss)
        throughput = len(train_windows) / epoch_duration

        print(
            f"[{model_name}] Epoch {epoch:02d}/{epochs:02d} | "
            f"Train MSE: {mean_loss:.6f} | "
            f"Time: {epoch_duration:.2f}s ({throughput:.1f} win/s)"
        )

    total_time = time.perf_counter() - start_total
    avg_throughput = total_samples / total_time
    param_count = sum(p.numel() for p in model.parameters())

    return {
        "model_name": model_name,
        "param_count": param_count,
        "total_train_time": total_time,
        "avg_epoch_time": float(np.mean(epoch_durations)),
        "avg_train_throughput": avg_throughput,
        "final_train_loss": epoch_losses[-1],
        "epoch_losses": epoch_losses,
    }


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    val_windows: np.ndarray,
    batch_size: int = 512,
    warmup_runs: int = 3,
    eval_runs: int = 5,
) -> Dict[str, float]:
    """Measures raw inference latency and reconstruction error statistics."""
    model.eval()
    tensor = to_tensor(val_windows).to(DEVICE)
    n_windows = len(val_windows)

    # Warm-up
    for _ in range(warmup_runs):
        _ = model(tensor[: min(batch_size, n_windows)])

    # Timed inference runs
    latencies = []
    errors = []

    for _ in range(eval_runs):
        start = time.perf_counter()
        run_errors = []
        for i in range(0, n_windows, batch_size):
            batch = tensor[i : i + batch_size]
            recon = model(batch)
            err = sequence_error(batch, recon)
            run_errors.append(err.cpu().numpy())
        latencies.append(time.perf_counter() - start)
        if len(errors) == 0:
            errors = np.concatenate(run_errors)

    avg_latency = float(np.mean(latencies))
    ms_per_window = (avg_latency / max(n_windows, 1)) * 1000.0
    inference_throughput = n_windows / avg_latency

    mean_err = float(errors.mean())
    std_err = float(errors.std())
    threshold = mean_err + THRESHOLD_K * std_err

    return {
        "ms_per_window": ms_per_window,
        "inference_throughput": inference_throughput,
        "val_reconstruction_mean": mean_err,
        "val_reconstruction_std": std_err,
        "calibrated_threshold": threshold,
    }


@torch.no_grad()
def score_sample_slice(
    model: nn.Module,
    scaler,
    df_slice: pd.DataFrame,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Scores a DataFrame slice for anomaly lead time evaluation."""
    model.eval()
    scaled = transform(df_slice, scaler)
    windows = build_windows(scaled, SEQUENCE_LENGTH, stride=1)
    timestamps = df_slice["timestamp"].values
    last_idx = np.arange(windows.shape[0]) + SEQUENCE_LENGTH - 1
    window_ts = timestamps[last_idx]

    tensor = to_tensor(windows).to(DEVICE)
    errs = []
    for i in range(0, len(tensor), batch_size):
        batch = tensor[i : i + batch_size]
        recon = model(batch)
        errs.append(sequence_error(batch, recon).cpu().numpy())
    seq_errs = np.concatenate(errs)

    scored = pd.DataFrame({"timestamp": window_ts, "seq_error": seq_errs})
    scored = scored.set_index("timestamp")
    scored["seq_error"] = scored["seq_error"].rolling("60s", min_periods=1).mean()
    return scored.reset_index()


def evaluate_single_failure_lead_time(
    scored: pd.DataFrame,
    threshold: float,
    failure: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculates lead time for a single specific failure event."""
    fail_start = pd.Timestamp(failure["start"])
    search_start = fail_start - pd.Timedelta(days=2)

    window = scored[(scored["timestamp"] >= search_start) & (scored["timestamp"] <= fail_start)]
    crossings = window[window["seq_error"] > threshold]

    if len(crossings) == 0:
        return {
            "failure_id": failure["id"],
            "failure_start": fail_start,
            "detected": False,
            "first_flag": None,
            "lead_time_minutes": None,
        }

    first_flag = crossings["timestamp"].iloc[0]
    lead_time = fail_start - first_flag
    lead_minutes = lead_time.total_seconds() / 60.0

    return {
        "failure_id": failure["id"],
        "failure_start": fail_start,
        "detected": True,
        "first_flag": first_flag,
        "lead_time_minutes": lead_minutes,
    }



def print_comparison_table(results: List[Dict[str, Any]]):
    """Prints a comparative summary table."""
    print("\n" + "=" * 90)
    print("                METROPT MODEL BENCHMARK RESULTS")
    print("=" * 90)

    header = (
        f"{'Metric':<36} | "
        f"{results[0]['model_name']:<24} | "
        f"{results[1]['model_name']:<24}"
    )
    print(header)
    print("-" * 90)

    rows = [
        ("Trainable Parameters", f"{results[0]['param_count']:,}", f"{results[1]['param_count']:,}"),
        ("Total Train Time (s)", f"{results[0]['total_train_time']:.2f}s", f"{results[1]['total_train_time']:.2f}s"),
        ("Avg Epoch Duration (s)", f"{results[0]['avg_epoch_time']:.2f}s", f"{results[1]['avg_epoch_time']:.2f}s"),
        ("Train Throughput (windows/s)", f"{results[0]['avg_train_throughput']:.1f} win/s", f"{results[1]['avg_train_throughput']:.1f} win/s"),
        ("Final Training Loss (MSE)", f"{results[0]['final_train_loss']:.6f}", f"{results[1]['final_train_loss']:.6f}"),
        ("Val Reconstruction MSE (mean)", f"{results[0]['val_reconstruction_mean']:.6f}", f"{results[1]['val_reconstruction_mean']:.6f}"),
        ("Val Reconstruction Std (sigma)", f"{results[0]['val_reconstruction_std']:.6f}", f"{results[1]['val_reconstruction_std']:.6f}"),
        ("Calibrated Threshold (mu+4*sig)", f"{results[0]['calibrated_threshold']:.6f}", f"{results[1]['calibrated_threshold']:.6f}"),
        ("Inference Latency (ms/window)", f"{results[0]['ms_per_window']:.4f} ms", f"{results[1]['ms_per_window']:.4f} ms"),
        ("Inference Throughput (win/s)", f"{results[0]['inference_throughput']:.1f} win/s", f"{results[1]['inference_throughput']:.1f} win/s"),
    ]

    for label, v1, v2 in rows:
        print(f"{label:<36} | {v1:<24} | {v2:<24}")

    print("=" * 90)


def run_benchmark(
    quick: bool = False,
    epochs: int = 5,
    alpha: float = 0.9,
    learnable_alpha: bool = True,
    stride: int = 30,
):
    print("=" * 90)
    print(f"Starting Side-by-Side Benchmark: Standard LSTM vs. Fractional LSTM (alpha={alpha})")
    print(f"Device: {DEVICE} | Quick Mode: {quick} | Epochs: {epochs}")
    print("=" * 90)

    healthy_train, stream = load_and_split()
    scaler = fit_scaler(healthy_train)

    if quick:
        print("\n[Benchmark] Quick mode active: taking a subset of healthy data for fast benchmarking.")
        healthy_subset = healthy_train.iloc[:15_000].reset_index(drop=True)
        val_subset = healthy_train.iloc[15_000:20_000].reset_index(drop=True)
    else:
        # Full validation split: last 5 days
        cutoff = healthy_train["timestamp"].iloc[-1] - pd.Timedelta(days=5)
        healthy_subset = healthy_train.loc[healthy_train["timestamp"] < cutoff].reset_index(drop=True)
        val_subset = healthy_train.loc[healthy_train["timestamp"] >= cutoff].reset_index(drop=True)

    scaled_train = transform(healthy_subset, scaler)
    train_windows = build_windows(scaled_train, SEQUENCE_LENGTH, stride=stride)

    scaled_val = transform(val_subset, scaler)
    val_windows = build_windows(scaled_val, SEQUENCE_LENGTH, stride=1)

    print(f"[Benchmark] Train Windows: {train_windows.shape} | Val Windows: {val_windows.shape}")

    # 1. Standard PyTorch LSTM Autoencoder
    std_model = LSTMAutoencoder().to(DEVICE)
    std_train_metrics = train_and_profile_model(
        std_model, "Standard LSTM AE", train_windows, epochs=epochs, lr=INITIAL_LR, batch_size=INITIAL_BATCH_SIZE
    )
    std_infer_metrics = benchmark_inference(std_model, val_windows)
    std_combined = {**std_train_metrics, **std_infer_metrics}

    # 2. Custom Fractional LSTM Autoencoder
    frac_model = FractionalLSTMAutoencoder(alpha=alpha, learnable_alpha=learnable_alpha).to(DEVICE)
    frac_train_metrics = train_and_profile_model(
        frac_model, f"Fractional LSTM AE (a={alpha})", train_windows, epochs=epochs, lr=INITIAL_LR, batch_size=INITIAL_BATCH_SIZE
    )
    frac_infer_metrics = benchmark_inference(frac_model, val_windows)
    frac_combined = {**frac_train_metrics, **frac_infer_metrics}

    # Compare and report
    print_comparison_table([std_combined, frac_combined])

    # Lead Time evaluation on Failure #1 (if available in stream)
    fail_1 = KNOWN_FAILURES[0]
    f1_start = pd.Timestamp(fail_1["start"])
    f1_window_start = f1_start - pd.Timedelta(days=1)
    f1_slice = stream.loc[(stream["timestamp"] >= f1_window_start) & (stream["timestamp"] <= f1_start + pd.Timedelta(hours=4))].reset_index(drop=True)

    if len(f1_slice) > SEQUENCE_LENGTH:
        print(f"\n--- Lead Time Evaluation on Known Failure #1 ({fail_1['type']} at {f1_start}) ---")
        scored_std = score_sample_slice(std_model, scaler, f1_slice)
        scored_frac = score_sample_slice(frac_model, scaler, f1_slice)

        std_lead = evaluate_single_failure_lead_time(scored_std, std_combined["calibrated_threshold"], fail_1)
        frac_lead = evaluate_single_failure_lead_time(scored_frac, frac_combined["calibrated_threshold"], fail_1)

        lead_std_val = std_lead["lead_time_minutes"] if std_lead["detected"] else "Not Detected"
        lead_frac_val = frac_lead["lead_time_minutes"] if frac_lead["detected"] else "Not Detected"

        print(f"Standard LSTM Lead Time:   {lead_std_val if isinstance(lead_std_val, str) else f'{lead_std_val:.1f} mins ({lead_std_val/60.0:.2f} hrs)'}")
        print(f"Fractional LSTM Lead Time: {lead_frac_val if isinstance(lead_frac_val, str) else f'{lead_frac_val:.1f} mins ({lead_frac_val/60.0:.2f} hrs)'}")


    print("\nBenchmark successfully completed!")
    return std_combined, frac_combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Side-by-side benchmark for MetroPT Autoencoders.")
    parser.add_argument("--quick", action="store_true", help="Run rapid benchmark on subset")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--alpha", type=float, default=0.9, help="Fractional alpha scaling")
    parser.add_argument("--learnable-alpha", action="store_true", default=True, help="Make alpha trainable")
    parser.add_argument("--stride", type=int, default=30, help="Window stride for training")
    args = parser.parse_args()

    run_benchmark(
        quick=args.quick,
        epochs=args.epochs,
        alpha=args.alpha,
        learnable_alpha=args.learnable_alpha,
        stride=args.stride,
    )
