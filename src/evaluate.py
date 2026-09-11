"""
Threshold calibration, dense stream scoring, lead-time evaluation, and
per-sensor explainability.

IMPORTANT MEMORY NOTE: densely scoring the full ~1.3M-row stream at
stride=1 as one giant window array would require materializing roughly
6+ GB of float32 data (1.3M windows x 180 timesteps x 7 features x 4 bytes).
To keep memory bounded on ordinary machines, `score_dataframe` processes
the stream in overlapping BLOCKS (each block is windowed, scored, and its
results appended to the output) rather than building one massive array.
This produces identical results to a naive whole-stream approach - same
per-timestep scores - just without the memory spike.
"""
import numpy as np
import pandas as pd
import torch

try:
    from src.config import (
        ANALOGUE_SENSORS, SEQUENCE_LENGTH, INFERENCE_STRIDE,
        THRESHOLD_K, ROLLING_SCORE_WINDOW,
        MODEL_PATH, SCALER_PATH, SCORED_STREAM_PATH,
        KNOWN_FAILURES, DEVICE,
    )
    from src.data_loader import load_and_split
    from src.windowing import load_scaler, transform, build_windows, to_tensor
    from src.model import LSTMAutoencoder, per_feature_error, sequence_error
except ImportError:
    from config import (
        ANALOGUE_SENSORS, SEQUENCE_LENGTH, INFERENCE_STRIDE,
        THRESHOLD_K, ROLLING_SCORE_WINDOW,
        MODEL_PATH, SCALER_PATH, SCORED_STREAM_PATH,
        KNOWN_FAILURES, DEVICE,
    )
    from data_loader import load_and_split
    from windowing import load_scaler, transform, build_windows, to_tensor
    from model import LSTMAutoencoder, per_feature_error, sequence_error

# Block size (in rows) for chunked dense scoring. Each block includes
# SEQUENCE_LENGTH-1 rows of overlap with the previous block so that no
# window is lost at the boundary.
SCORE_BLOCK_ROWS = 20_000


def load_model(path: str = MODEL_PATH) -> tuple:
    """Returns (model, baseline_error, chunk_log)."""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["model_config"]
    model = LSTMAutoencoder(
        n_features=cfg["n_features"], hidden_size=cfg["hidden_size"],
        latent_size=cfg["latent_size"], num_layers=cfg["num_layers"],
        seq_len=cfg["seq_len"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["baseline_error"], checkpoint.get("chunk_log", [])


@torch.no_grad()
def calibrate_threshold(model, healthy_train: pd.DataFrame, scaler,
                         val_days: int = 5) -> dict:
    """Carve the last `val_days` of healthy_train as a validation slice,
    compute mu + THRESHOLD_K * sigma of its dense reconstruction error."""
    cutoff = healthy_train["timestamp"].iloc[-1] - pd.Timedelta(days=val_days)
    val_slice = healthy_train.loc[healthy_train["timestamp"] >= cutoff].reset_index(drop=True)

    scaled = transform(val_slice, scaler)
    windows = build_windows(scaled, SEQUENCE_LENGTH, INFERENCE_STRIDE)
    tensor = to_tensor(windows).to(DEVICE)

    errors = []
    for i in range(0, len(tensor), 512):
        batch = tensor[i:i + 512]
        recon = model(batch)
        errors.append(sequence_error(batch, recon).cpu().numpy())
    errors = np.concatenate(errors)

    mean, std = float(errors.mean()), float(errors.std())
    threshold = mean + THRESHOLD_K * std
    print(f"[threshold] healthy mean={mean:.6f} std={std:.6f} "
          f"-> threshold={threshold:.6f} (k={THRESHOLD_K})")
    return {"mean": mean, "std": std, "threshold": threshold}


@torch.no_grad()
def score_dataframe(model, scaler, df: pd.DataFrame, batch_size: int = 512,
                     progress_callback=None) -> pd.DataFrame:
    """Dense (stride=1) scoring of `df`, processed in memory-bounded
    overlapping blocks. Returns a DataFrame with columns:
    timestamp, seq_error, <sensor>_error for each analogue sensor.
    """
    progress_callback = progress_callback or (lambda e: None)
    overlap = SEQUENCE_LENGTH - 1
    n_rows = len(df)
    results = []

    block_start = 0
    while block_start < n_rows:
        block_end = min(block_start + SCORE_BLOCK_ROWS, n_rows)
        # extend the block backwards by `overlap` rows (except for the very
        # first block) so windows spanning the block boundary aren't lost.
        read_start = max(0, block_start - overlap) if block_start > 0 else 0
        block_df = df.iloc[read_start:block_end].reset_index(drop=True)

        if len(block_df) < SEQUENCE_LENGTH:
            block_start = block_end
            continue

        scaled = transform(block_df, scaler)
        windows = build_windows(scaled, SEQUENCE_LENGTH, INFERENCE_STRIDE)

        timestamps = block_df["timestamp"].values
        n_windows = windows.shape[0]
        last_idx = np.arange(n_windows) * INFERENCE_STRIDE + SEQUENCE_LENGTH - 1
        window_ts = timestamps[last_idx]

        tensor = to_tensor(windows).to(DEVICE)
        seq_errs, feat_errs = [], []
        for i in range(0, len(tensor), batch_size):
            batch = tensor[i:i + batch_size]
            recon = model(batch)
            seq_errs.append(sequence_error(batch, recon).cpu().numpy())
            feat_errs.append(per_feature_error(batch, recon).cpu().numpy())
        seq_errs = np.concatenate(seq_errs)
        feat_errs = np.concatenate(feat_errs, axis=0)

        block_result = pd.DataFrame({"timestamp": window_ts, "seq_error": seq_errs})
        for i, sensor in enumerate(ANALOGUE_SENSORS):
            block_result[f"{sensor}_error"] = feat_errs[:, i]

        # If this wasn't the first block, the overlapping rows produce
        # windows whose LAST timestep falls before `block_start`'s actual
        # timestamp - drop those duplicates (already covered by the
        # previous block's output).
        if block_start > 0:
            cutoff_ts = df["timestamp"].iloc[block_start]
            block_result = block_result[block_result["timestamp"] >= cutoff_ts]

        results.append(block_result)
        progress_callback({
            "rows_scored": block_end, "total_rows": n_rows,
            "pct": round(100 * block_end / n_rows, 1),
        })
        print(f"[scoring] processed rows {block_start:,}-{block_end:,} / {n_rows:,} "
              f"({100*block_end/n_rows:.1f}%)")

        block_start = block_end

    scored = pd.concat(results, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    # 60-second rolling mean smoothing on the scalar score.
    scored = scored.set_index("timestamp")
    scored["seq_error"] = scored["seq_error"].rolling(f"{ROLLING_SCORE_WINDOW}s", min_periods=1).mean()
    scored = scored.reset_index()

    return scored


def evaluate_lead_times(scored: pd.DataFrame, threshold: float) -> list:
    """For each known failure, find the first threshold-crossing within 2
    days before the failure's documented start."""
    results = []
    for f in KNOWN_FAILURES:
        fail_start = pd.Timestamp(f["start"])
        search_start = fail_start - pd.Timedelta(days=2)

        window = scored[(scored["timestamp"] >= search_start) & (scored["timestamp"] <= fail_start)]
        crossings = window[window["seq_error"] > threshold]

        if len(crossings) == 0:
            results.append({
                "failure_id": f["id"], "failure_start": fail_start,
                "detected": False, "first_flag": None, "lead_time_minutes": None,
            })
            print(f"Failure #{f['id']} ({fail_start}): NOT DETECTED within 2-day lookback window")
        else:
            first_flag = crossings["timestamp"].iloc[0]
            lead_time = fail_start - first_flag
            lead_minutes = lead_time.total_seconds() / 60
            results.append({
                "failure_id": f["id"], "failure_start": fail_start,
                "detected": True, "first_flag": first_flag,
                "lead_time_minutes": lead_minutes,
            })
            print(f"Failure #{f['id']} ({fail_start}): first flagged at {first_flag} "
                  f"— lead time = {lead_minutes:.1f} min")
    return results


def top_contributing_sensors(scored: pd.DataFrame, timestamp: pd.Timestamp, top_n: int = 3) -> list:
    """Rank analogue sensors by per-feature error at (or nearest to) `timestamp`."""
    idx = (scored["timestamp"] - timestamp).abs().idxmin()
    row = scored.loc[idx]
    error_cols = [f"{s}_error" for s in ANALOGUE_SENSORS]
    ranked = row[error_cols].sort_values(ascending=False)
    return [(col, float(val)) for col, val in ranked.head(top_n).items()]


def main():
    model, baseline_error, chunk_log = load_model()
    scaler = load_scaler()
    healthy_train, stream = load_and_split()

    threshold_info = calibrate_threshold(model, healthy_train, scaler)
    threshold = threshold_info["threshold"]

    print("[scoring] dense pass over the full stream (this can take a while)...")
    scored = score_dataframe(model, scaler, stream)
    scored.to_csv(SCORED_STREAM_PATH, index=False)
    print(f"[scoring] Saved {len(scored):,} scored rows to {SCORED_STREAM_PATH}")

    print("\n=== Detection lead time vs. known failures ===")
    lead_time_results = evaluate_lead_times(scored, threshold)

    print("\n=== Example explanation for each detected failure's first flag ===")
    for r in lead_time_results:
        if r["detected"]:
            top_sensors = top_contributing_sensors(scored, r["first_flag"])
            print(f"Failure #{r['failure_id']} near {r['failure_start']}: top sensors -> {top_sensors}")

    return threshold_info, scored, lead_time_results


if __name__ == "__main__":
    main()
