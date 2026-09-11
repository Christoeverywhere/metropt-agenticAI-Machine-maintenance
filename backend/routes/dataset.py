"""
Dataset routes — upload CSV and get summary statistics.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

# Add src/ to path for pipeline imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
import data_loader

from backend.schemas import (
    DatasetSummaryResponse,
    DatasetUploadResponse,
    SensorMissing,
)

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
    """Accept a CSV upload, validate schema, return row count and date range."""
    try:
        # Read uploaded file into memory, then save to data/
        content = await file.read()
        csv_path = Path(config.RAW_CSV_PATH)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_bytes(content)

        # Validate by loading
        df = data_loader.load_raw_data(str(csv_path))

        return DatasetUploadResponse(
            status="ok",
            row_count=len(df),
            date_start=str(df[config.TIMESTAMP_COL].min()),
            date_end=str(df[config.TIMESTAMP_COL].max()),
            columns=list(df.columns),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/summary", response_model=DatasetSummaryResponse)
async def dataset_summary() -> DatasetSummaryResponse:
    """Return basic stats about the loaded dataset."""
    if not Path(config.RAW_CSV_PATH).exists():
        raise HTTPException(
            status_code=404,
            detail="No dataset found. Upload or place the CSV in data/.",
        )

    try:
        df = data_loader.load_raw_data()
        healthy, stream = data_loader.split_healthy_and_stream(df)

        all_sensors = config.ANALOGUE_SENSORS + config.DIGITAL_SENSORS
        missing_vals = []
        for sensor in all_sensors:
            if sensor in df.columns:
                n_miss = int(df[sensor].isna().sum())
                pct = round(n_miss / len(df) * 100, 2)
                missing_vals.append(
                    SensorMissing(sensor=sensor, missing_count=n_miss, missing_pct=pct)
                )

        return DatasetSummaryResponse(
            row_count=len(df),
            date_start=str(df[config.TIMESTAMP_COL].min()),
            date_end=str(df[config.TIMESTAMP_COL].max()),
            healthy_train_rows=len(healthy),
            stream_rows=len(stream),
            missing_values=missing_vals,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Summary failed: {str(e)}")
