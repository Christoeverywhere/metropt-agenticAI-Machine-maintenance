"""
Evaluate the trained autoencoder:

  1. Calibrate an anomaly threshold from the healthy training error distribution.
  2. Score the full stream densely (stride=1) and smooth with a rolling window.
  3. For each of the 4 known failures, find the first thresh old-crossing
     BEFORE the documented failure start, and report the lead time —
     this is what you benchmark against the published 97min–16hr range.
  4. For any flagged window, report per-sensor reconstruction error so you
     can say *which* sensor is driving the anomaly (the agent's "why").

Run:
    python evaluate.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch

import config
import data_loader
import windowing
from model import LSTMAutoencoder, per_feature_reconstruction_error, sequence_reconstruction_error

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_scaler():
    model = LSTMAutoencoder(n_features=len(config.ANALOGUE_FEATURES)).to(DEVICE)
    model.load_state_dict(torch.load(config.MODEL_CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    with open(config.SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def score_dataframe(model, scaler, df: pd.DataFrame) -> pd.DataFrame:
    """
    Dense per-timestep scoring. Returns a dataframe indexed by the ORIGINAL
    row index of `df`, with columns:
        timestamp, seq_error, <feature>_error for each analogue feature
    Each sequence's error is attributed to its LAST timestep (the most
    "current" reading at inference time — matches how you'd score a live stream).
    """
    scaled = windowing.scale(df, scaler)
    sequences = windowing.make_sequences(scaled, config.SEQUENCE_LENGTH, config.INFERENCE_STRIDE)
    starts = windowing.sequence_start_indices(
        len(df), config.SEQUENCE_LENGTH, config.INFERENCE_STRIDE
    )
    if len(sequences) == 0:
        return pd.DataFrame(columns=["timestamp", "seq_error"] +
                             [f"{f}_error" for f in config.ANALOGUE_FEATURES])

    batch_size = 512
    seq_errors, feat_errors = [], []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(sequences[i : i + batch_size]).to(DEVICE)
            x_hat = model(batch)
            seq_errors.append(sequence_reconstruction_error(batch, x_hat).cpu().numpy())
            feat_errors.append(per_feature_reconstruction_error(batch, x_hat).cpu().numpy())

    seq_errors = np.concatenate(seq_errors)
    feat_errors = np.concatenate(feat_errors, axis=0)

    last_step_idx = starts + config.SEQUENCE_LENGTH - 1
    result = pd.DataFrame({
        "timestamp": df[config.TIMESTAMP_COL].values[last_step_idx],
        "seq_error": seq_errors,
    })
    for j, feat in enumerate(config.ANALOGUE_FEATURES):
        result[f"{feat}_error"] = feat_errors[:, j]

    return result


def calibrate_threshold(model, scaler, healthy_val_df: pd.DataFrame) -> float:
    scored = score_dataframe(model, scaler, healthy_val_df)
    mu, sigma = scored["seq_error"].mean(), scored["seq_error"].std()
    threshold = mu + config.THRESHOLD_K * sigma
    print(f"[threshold] healthy mean={mu:.6f} std={sigma:.6f} "
          f"-> threshold={threshold:.6f} (k={config.THRESHOLD_K})")
    return threshold


def evaluate_lead_times(scored: pd.DataFrame, threshold: float):
    scored = scored.sort_values("timestamp").reset_index(drop=True)
    scored["rolling_score"] = (
        scored["seq_error"].rolling(config.ROLLING_SCORE_WINDOW, min_periods=1).mean()
    )
    scored["flagged"] = scored["rolling_score"] > threshold

    print("\n=== Detection lead time vs. known failures ===")
    for f in config.KNOWN_FAILURES:
        fail_start = pd.Timestamp(f["start"])
        before = scored[scored["timestamp"] < fail_start]
        crossings = before[before["flagged"]]

        if crossings.empty:
            print(f"Failure #{f['id']} ({f['start']}): NOT detected before onset")
            continue

        first_flag_time = crossings["timestamp"].iloc[0]
        # last contiguous flagged window ending right before fail_start,
        # to avoid reporting a stale flag from long before as "lead time"
        contiguous = crossings[crossings["timestamp"] >= fail_start - pd.Timedelta(days=2)]
        first_flag_time = contiguous["timestamp"].iloc[0] if not contiguous.empty else first_flag_time

        lead = fail_start - first_flag_time
        print(f"Failure #{f['id']} ({f['start']}): first flagged at "
              f"{first_flag_time} — lead time = {lead}")

    return scored


def top_contributing_sensors(scored: pd.DataFrame, timestamp: pd.Timestamp, top_k: int = 3):
    """
    The agent's 'why' explanation for a flagged moment: rank sensors by
    reconstruction error at (or nearest to) the given timestamp.
    """
    row = scored.iloc[(scored["timestamp"] - timestamp).abs().argsort()[:1]]
    error_cols = [f"{feat}_error" for feat in config.ANALOGUE_FEATURES]
    ranked = row[error_cols].iloc[0].sort_values(ascending=False)
    return list(ranked.head(top_k).items())


def main():
    model, scaler = load_model_and_scaler()

    df = data_loader.load_raw()
    healthy_train_df, stream_df = data_loader.split_healthy_train_and_stream(df)

    # Hold out the tail of the healthy month as a validation slice for
    # threshold calibration (avoids calibrating on exactly what we trained on).
    val_cutoff = healthy_train_df[config.TIMESTAMP_COL].iloc[-1] - pd.Timedelta(days=5)
    healthy_val_df = healthy_train_df[healthy_train_df[config.TIMESTAMP_COL] >= val_cutoff]

    threshold = calibrate_threshold(model, scaler, healthy_val_df)

    print("\n[scoring] dense pass over the full stream (this can take a while)...")
    scored = score_dataframe(model, scaler, stream_df)
    scored = evaluate_lead_times(scored, threshold)

    print("\n=== Example explanation for each detected failure's first flag ===")
    for f in config.KNOWN_FAILURES:
        fail_start = pd.Timestamp(f["start"])
        top = top_contributing_sensors(scored, fail_start)
        print(f"Failure #{f['id']} near {fail_start}: top sensors -> {top}")

    scored.to_csv(config.CHECKPOINT_DIR / "scored_stream.csv", index=False)
    print(f"\nSaved scored stream -> {config.CHECKPOINT_DIR / 'scored_stream.csv'}")


if __name__ == "__main__":
    main()
