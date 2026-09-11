"""
Model status route — whether a trained model exists, when it was last trained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from fastapi import APIRouter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config

from backend.schemas import ModelStatusResponse

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/status", response_model=ModelStatusResponse)
async def model_status() -> ModelStatusResponse:
    """Check whether a trained model checkpoint exists and return metadata."""
    model_exists = Path(config.MODEL_PATH).exists()
    scaler_exists = Path(config.SCALER_PATH).exists()
    scored_exists = Path(config.SCORED_STREAM_PATH).exists()

    last_trained = None
    baseline_error = None

    if model_exists:
        try:
            checkpoint = torch.load(
                config.MODEL_PATH,
                map_location="cpu",
                weights_only=False,
            )
            baseline_error = checkpoint.get("baseline_error")
            last_trained = checkpoint.get("saved_at")
        except Exception:
            pass

    return ModelStatusResponse(
        exists=model_exists,
        last_trained=last_trained,
        baseline_error=baseline_error,
        scaler_exists=scaler_exists,
        scored_stream_exists=scored_exists,
    )
