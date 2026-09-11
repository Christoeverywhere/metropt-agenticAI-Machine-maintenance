"""
Score and sensor data routes — paginated/downsampled score access,
raw sensor readings, and known failure windows.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
import data_loader

from backend.schemas import (
    FailuresResponse,
    FailureWindow,
    ScoresResponse,
    SensorDataResponse,
)

router = APIRouter(prefix="/api", tags=["scores"])

# Maximum points returned in a single response to avoid overwhelming the client
MAX_POINTS = 2000


def _downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Simple stride-based downsampling to keep response size manageable."""
    if len(df) <= max_points:
        return df
    stride = max(1, len(df) // max_points)
    return df.iloc[::stride].reset_index(drop=True)


@router.get("/scores", response_model=ScoresResponse)
async def get_scores(
    start: Optional[str] = Query(None, description="ISO timestamp start"),
    end: Optional[str] = Query(None, description="ISO timestamp end"),
    stride: Optional[int] = Query(None, description="Custom stride for downsampling"),
    max_points: int = Query(MAX_POINTS, description="Max points to return"),
) -> ScoresResponse:
    """Return a paginated/downsampled slice of scored_stream.csv for charting."""
    if not Path(config.SCORED_STREAM_PATH).exists():
        raise HTTPException(
            status_code=404,
            detail="No scored stream found. Run evaluation first.",
        )

    df = pd.read_csv(config.SCORED_STREAM_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    total_rows = len(df)

    # Filter by time range if specified
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end)]

    # Downsample
    if stride and stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    else:
        df = _downsample(df, max_points)

    # Convert timestamps to ISO strings for JSON serialization
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    points = df.to_dict(orient="records")

    return ScoresResponse(
        points=points,
        total_rows=total_rows,
        returned_rows=len(points),
        start=str(df["timestamp"].iloc[0]) if len(df) > 0 else "",
        end=str(df["timestamp"].iloc[-1]) if len(df) > 0 else "",
    )


@router.get("/sensors/raw", response_model=SensorDataResponse)
async def get_raw_sensor(
    sensor: str = Query(..., description="Sensor column name"),
    start: Optional[str] = Query(None, description="ISO timestamp start"),
    end: Optional[str] = Query(None, description="ISO timestamp end"),
    scaled: bool = Query(False, description="Return scaled (normalised) values"),
    max_points: int = Query(MAX_POINTS, description="Max points to return"),
) -> SensorDataResponse:
    """Return raw (or scaled) sensor readings for a time range."""
    all_sensors = config.ANALOGUE_SENSORS + config.DIGITAL_SENSORS
    if sensor not in all_sensors:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown sensor '{sensor}'. Valid: {all_sensors}",
        )

    if not Path(config.RAW_CSV_PATH).exists():
        raise HTTPException(status_code=404, detail="No dataset found.")

    df = data_loader.load_raw_data()
    total_rows = len(df)

    # Filter by time range
    if start:
        df = df[df[config.TIMESTAMP_COL] >= pd.Timestamp(start)]
    if end:
        df = df[df[config.TIMESTAMP_COL] <= pd.Timestamp(end)]

    # Optionally apply scaler (only for analogue sensors)
    if scaled and sensor in config.ANALOGUE_SENSORS and Path(config.SCALER_PATH).exists():
        import windowing
        scaler = windowing.load_scaler()
        # Scale all analogue features, then extract the one we need
        scaled_vals = scaler.transform(df[config.ANALOGUE_SENSORS].values)
        idx = config.ANALOGUE_SENSORS.index(sensor)
        values = scaled_vals[:, idx]
    else:
        values = df[sensor].values

    # Build result DataFrame for downsampling
    result_df = pd.DataFrame({
        "timestamp": df[config.TIMESTAMP_COL].values,
        "value": values,
    })
    result_df = _downsample(result_df, max_points)
    result_df["timestamp"] = pd.to_datetime(result_df["timestamp"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    points = result_df.to_dict(orient="records")

    return SensorDataResponse(
        sensor=sensor,
        scaled=scaled,
        points=points,
        total_rows=total_rows,
        returned_rows=len(points),
    )


@router.get("/failures", response_model=FailuresResponse)
async def get_failures() -> FailuresResponse:
    """Return the 4 known failure windows from config."""
    failures = [
        FailureWindow(
            id=f["id"],
            start=f["start"],
            end=f["end"],
            type=f["type"],
            severity=f["severity"],
        )
        for f in config.KNOWN_FAILURES
    ]
    return FailuresResponse(failures=failures)
