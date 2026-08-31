"""
Load the MetroPT-3 CSV, parse timestamps, and produce the splits used by
the training pipeline:

  - "healthy_train": first month of data (per the UCI recommended split),
    used to fit the scaler and train the initial autoencoder.
  - "stream": everything after the first month, walked chunk-by-chunk to
    simulate batches "arriving" for incremental fine-tuning + scoring.

Failure windows are never used as training labels — we're training an
autoencoder on (presumed) normal behaviour only. They're used purely to
(a) exclude obviously-anomalous rows from fine-tuning and (b) evaluate
detection lead time.
"""

from __future__ import annotations

import pandas as pd

import config


def load_raw(csv_path=config.RAW_CSV_PATH) -> pd.DataFrame:
    """Load the raw CSV and normalise column names / dtypes."""
    df = pd.read_csv(csv_path)

    # The published CSV sometimes ships an unnamed index column and slightly
    # different casing/spacing on headers depending on export — normalise.
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    if "Unnamed:_0" in df.columns:
        df = df.drop(columns=["Unnamed:_0"])

    if config.TIMESTAMP_COL not in df.columns:
        raise ValueError(
            f"Expected a '{config.TIMESTAMP_COL}' column, found: {list(df.columns)}"
        )

    df[config.TIMESTAMP_COL] = pd.to_datetime(df[config.TIMESTAMP_COL])
    df = df.sort_values(config.TIMESTAMP_COL).reset_index(drop=True)

    missing = [c for c in config.ANALOGUE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected analogue columns: {missing}")

    return df


def known_failure_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask, True for any row that falls inside a documented failure window."""
    mask = pd.Series(False, index=df.index)
    for f in config.KNOWN_FAILURES:
        start, end = pd.Timestamp(f["start"]), pd.Timestamp(f["end"])
        mask |= (df[config.TIMESTAMP_COL] >= start) & (df[config.TIMESTAMP_COL] <= end)
    return mask


def split_healthy_train_and_stream(df: pd.DataFrame):
    """
    First calendar month -> healthy_train (used for initial fit).
    Remainder -> stream (walked in chunks for incremental fine-tuning/scoring).

    Rows inside a known failure window are dropped from healthy_train even
    though none are expected in month 1 — this is a safety net in case the
    data you're given starts earlier/later than assumed.
    """
    t0 = df[config.TIMESTAMP_COL].iloc[0]
    month_cutoff = t0 + pd.DateOffset(months=1)

    fail_mask = known_failure_mask(df)

    healthy_train = df[(df[config.TIMESTAMP_COL] < month_cutoff) & (~fail_mask)].copy()
    stream = df[df[config.TIMESTAMP_COL] >= month_cutoff].copy()

    return healthy_train.reset_index(drop=True), stream.reset_index(drop=True)


def iter_stream_chunks(stream_df: pd.DataFrame, freq: str = config.FINETUNE_CHUNK):
    """
    Yield (chunk_start, chunk_end, chunk_df) tuples walking the stream in
    fixed-size time chunks — this is what simulates "batch by batch" arrival
    of new data for incremental fine-tuning.
    """
    if stream_df.empty:
        return

    t0 = stream_df[config.TIMESTAMP_COL].iloc[0]
    t_end = stream_df[config.TIMESTAMP_COL].iloc[-1]

    bounds = pd.date_range(start=t0, end=t_end + pd.Timedelta(freq), freq=freq)
    for start, end in zip(bounds[:-1], bounds[1:]):
        chunk = stream_df[
            (stream_df[config.TIMESTAMP_COL] >= start)
            & (stream_df[config.TIMESTAMP_COL] < end)
        ]
        if len(chunk) > 0:
            yield start, end, chunk.reset_index(drop=True)