"""
Loading, cleaning, and splitting the MetroPT-3 dataset.

Responsibilities:
  1. Robustly load the ~218MB CSV.
  2. Normalize column names (the raw header has a couple of quirks).
  3. Parse timestamps, sort chronologically, sanity-check the result.
  4. Split into:
       - healthy_train: first calendar month, with any rows inside a known
         failure window stripped out as a safety net.
       - stream: everything after month 1 (~5 months), used for Stage 2
         incremental fine-tuning and for evaluation.
"""
import pandas as pd

try:
    from src.config import (
        RAW_CSV_PATH,
        RAW_TO_STANDARD_COLUMN_MAP,
        TIMESTAMP_COL,
        ALL_SENSORS,
        KNOWN_FAILURES,
    )
except ImportError:
    from config import (
        RAW_CSV_PATH,
        RAW_TO_STANDARD_COLUMN_MAP,
        TIMESTAMP_COL,
        ALL_SENSORS,
        KNOWN_FAILURES,
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, drop unnamed index columns, apply the known
    raw->standard renames (e.g. 'DV_eletric' -> 'DV_electric')."""
    df = df.copy()

    # Drop pandas' auto-generated unnamed index column(s), if present.
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Normalize whitespace in column names.
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    # Apply known raw -> standard renames.
    rename_map = {k: v for k, v in RAW_TO_STANDARD_COLUMN_MAP.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def load_raw_data(csv_path: str = RAW_CSV_PATH) -> pd.DataFrame:
    """Load the raw CSV, normalize columns/timestamps, sort chronologically,
    and run sanity checks. Raises clear errors rather than failing silently
    or downstream in some unrelated function."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find dataset at '{csv_path}'. Download the MetroPT-3 "
            f"CSV and place it at this path (see README > Data Setup)."
        ) from e

    df = _normalize_columns(df)

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(
            f"Expected a '{TIMESTAMP_COL}' column after normalization, "
            f"got columns: {list(df.columns)}"
        )

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # --- Sanity checks -----------------------------------------------------
    missing_sensors = [s for s in ALL_SENSORS if s not in df.columns]
    if missing_sensors:
        raise ValueError(
            f"Dataset is missing expected sensor columns after normalization: "
            f"{missing_sensors}. Available columns: {list(df.columns)}"
        )

    n_dupes = df[TIMESTAMP_COL].duplicated().sum()
    if n_dupes > 0:
        raise ValueError(
            f"Found {n_dupes} duplicate timestamps in the dataset. "
            f"Expected a clean 1-row-per-timestamp series."
        )

    if not df[TIMESTAMP_COL].is_monotonic_increasing:
        raise ValueError("Timestamps are not monotonically increasing after sorting.")

    fully_null_sensors = [s for s in ALL_SENSORS if df[s].isnull().all()]
    if fully_null_sensors:
        raise ValueError(f"These sensor columns are entirely null: {fully_null_sensors}")

    return df


def _strip_known_failure_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any rows that fall inside a documented failure window. This is
    a safety net for healthy_train even though month 1 (Feb 2020) predates
    all 4 known failures (which occur Apr-Jul 2020) — kept generic in case
    the known-failures list or split boundary ever changes."""
    mask = pd.Series(False, index=df.index)
    for f in KNOWN_FAILURES:
        start, end = pd.Timestamp(f["start"]), pd.Timestamp(f["end"])
        mask |= (df[TIMESTAMP_COL] >= start) & (df[TIMESTAMP_COL] <= end)
    n_removed = mask.sum()
    if n_removed > 0:
        print(f"[data_loader] Stripped {n_removed} rows inside known failure windows from healthy_train.")
    return df.loc[~mask].reset_index(drop=True)


def split_healthy_and_stream(df: pd.DataFrame):
    """Split into (healthy_train, stream).

    healthy_train = first calendar month from the first timestamp in the
                     data, with any known-failure-window rows stripped out.
    stream        = everything from the second calendar month onward.
    """
    first_ts = df[TIMESTAMP_COL].iloc[0]
    month1_start = first_ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    # first day of the following month
    if month1_start.month == 12:
        month1_end = month1_start.replace(year=month1_start.year + 1, month=1)
    else:
        month1_end = month1_start.replace(month=month1_start.month + 1)

    month1_mask = (df[TIMESTAMP_COL] >= month1_start) & (df[TIMESTAMP_COL] < month1_end)

    healthy_train = df.loc[month1_mask].reset_index(drop=True)
    stream = df.loc[~month1_mask].reset_index(drop=True)

    healthy_train = _strip_known_failure_windows(healthy_train)

    if len(healthy_train) == 0:
        raise ValueError("healthy_train is empty after splitting/cleaning - check the data's date range.")
    if len(stream) == 0:
        raise ValueError("stream is empty after splitting - dataset may only cover a single month.")

    return healthy_train, stream


def load_and_split(csv_path: str = RAW_CSV_PATH):
    """Convenience wrapper: load, clean, and split in one call."""
    df = load_raw_data(csv_path)
    healthy_train, stream = split_healthy_and_stream(df)
    print(f"[data_loader] Loaded {len(df):,} total rows "
          f"({df[TIMESTAMP_COL].min()} -> {df[TIMESTAMP_COL].max()})")
    print(f"[data_loader] healthy_train: {len(healthy_train):,} rows "
          f"({healthy_train[TIMESTAMP_COL].min()} -> {healthy_train[TIMESTAMP_COL].max()})")
    print(f"[data_loader] stream: {len(stream):,} rows "
          f"({stream[TIMESTAMP_COL].min()} -> {stream[TIMESTAMP_COL].max()})")
    return healthy_train, stream


if __name__ == "__main__":
    load_and_split()
