"""
Preprocessing, Cleaning, Missing-Value Handling, and Temporal Partitioning
for the MetroPT-3 Dataset with Zero Data Leakage.
"""
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from src.config import (
        RAW_CSV_PATH,
        DATA_DIR,
        RAW_TO_STANDARD_COLUMN_MAP,
        TIMESTAMP_COL,
        FEATURES,
        ANALOGUE_SENSORS,
        DIGITAL_SENSORS,
        KNOWN_FAILURES,
    )
except ImportError:
    from config import (
        RAW_CSV_PATH,
        DATA_DIR,
        RAW_TO_STANDARD_COLUMN_MAP,
        TIMESTAMP_COL,
        FEATURES,
        ANALOGUE_SENSORS,
        DIGITAL_SENSORS,
        KNOWN_FAILURES,
    )


def find_csv_path() -> str:
    """Finds the MetroPT3 CSV file in candidate locations."""
    candidates = [
        RAW_CSV_PATH,
        os.path.join(DATA_DIR, "MetroPT3(AirCompressor).csv"),
        os.path.join(os.path.dirname(DATA_DIR), "MetroPT3(AirCompressor).csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"MetroPT3(AirCompressor).csv not found in any candidate path: {candidates}")


def load_and_clean_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Loads raw MetroPT-3 data, standardizes columns, handles missing values,
    and sorts chronologically."""
    path = csv_path or find_csv_path()
    print(f"[preprocessing] Loading raw CSV from: {path}")
    df = pd.read_csv(path)

    # 1. Drop auto-generated unnamed index columns
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # 2. Normalize whitespace in column names
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    # 3. Apply raw -> standard column renames
    rename_map = {k: v for k, v in RAW_TO_STANDARD_COLUMN_MAP.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    # 4. Parse timestamp and sort strictly chronologically
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # 5. Handle missing values: forward fill then backward fill for continuous sensor physics
    n_nans = df[ANALOGUE_SENSORS].isna().sum().sum()
    if n_nans > 0:
        print(f"[preprocessing] Found {n_nans} missing values in analogue sensors. Applying forward/backward fill.")
        df[ANALOGUE_SENSORS] = df[ANALOGUE_SENSORS].ffill().bfill()

    print(f"[preprocessing] Cleaned dataset: {len(df):,} rows from {df[TIMESTAMP_COL].iloc[0]} to {df[TIMESTAMP_COL].iloc[-1]}")
    return df


def create_temporal_splits(
    df: pd.DataFrame,
    val_days: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits data strictly along the temporal axis to guarantee ZERO data leakage:
      - healthy_train: 2020-02-01 to 2020-02-23 (Normal baseline operation)
      - healthy_val:   2020-02-24 to 2020-02-28 (Normal validation for early stopping & threshold calibration)
      - test_stream:   2020-03-01 to 2020-09-01 (Continuous stream containing normal operations + 4 failures)
    """
    month_1_end = pd.Timestamp("2020-02-28 23:59:59")
    stream_start = pd.Timestamp("2020-03-01 00:00:00")

    # Safety check: exclude any known failure time intervals from month 1 (healthy month)
    m1_mask = df[TIMESTAMP_COL] <= month_1_end
    for f in KNOWN_FAILURES:
        f_start, f_end = pd.Timestamp(f["start"]), pd.Timestamp(f["end"])
        m1_mask &= ~((df[TIMESTAMP_COL] >= f_start) & (df[TIMESTAMP_COL] <= f_end))

    month_1 = df.loc[m1_mask].reset_index(drop=True)
    test_stream = df.loc[df[TIMESTAMP_COL] >= stream_start].reset_index(drop=True)

    # Partition month 1 into Train and Validation
    val_cutoff = month_1[TIMESTAMP_COL].iloc[-1] - pd.Timedelta(days=val_days)
    healthy_train = month_1.loc[month_1[TIMESTAMP_COL] < val_cutoff].reset_index(drop=True)
    healthy_val = month_1.loc[month_1[TIMESTAMP_COL] >= val_cutoff].reset_index(drop=True)

    print(f"[splits] healthy_train: {len(healthy_train):,} rows ({healthy_train[TIMESTAMP_COL].iloc[0].date()} -> {healthy_train[TIMESTAMP_COL].iloc[-1].date()})")
    print(f"[splits] healthy_val:   {len(healthy_val):,} rows ({healthy_val[TIMESTAMP_COL].iloc[0].date()} -> {healthy_val[TIMESTAMP_COL].iloc[-1].date()})")
    print(f"[splits] test_stream:   {len(test_stream):,} rows ({test_stream[TIMESTAMP_COL].iloc[0].date()} -> {test_stream[TIMESTAMP_COL].iloc[-1].date()})")

    return healthy_train, healthy_val, test_stream


def fit_scaler(healthy_train: pd.DataFrame) -> StandardScaler:
    """Fits StandardScaler ONLY on healthy_train analogue sensors to prevent data leakage."""
    scaler = StandardScaler()
    scaler.fit(healthy_train[ANALOGUE_SENSORS].values)
    return scaler


def transform_data(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Applies fitted scaler to a DataFrame. Returns float32 array (n_rows, n_features)."""
    return scaler.transform(df[ANALOGUE_SENSORS].values).astype(np.float32)


def generate_ground_truth_labels(
    df: pd.DataFrame,
    failures: List[Dict] = KNOWN_FAILURES,
    pre_failure_hours: float = 12.0,
) -> np.ndarray:
    """Generates ground truth binary labels (0=Normal, 1=Anomaly) for evaluation.
    Flags both the documented failure interval and the immediate pre-failure degradation window.
    """
    labels = np.zeros(len(df), dtype=np.int32)
    timestamps = pd.to_datetime(df[TIMESTAMP_COL]).values

    for f in failures:
        f_start = np.datetime64(pd.Timestamp(f["start"]) - pd.Timedelta(hours=pre_failure_hours))
        f_end = np.datetime64(pd.Timestamp(f["end"]))
        mask = (timestamps >= f_start) & (timestamps <= f_end)
        labels[mask] = 1

    return labels
