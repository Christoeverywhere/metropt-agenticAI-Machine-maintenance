"""
Training routes — start training, poll status, SSE live stream.
"""

from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train as train_module

from backend.job_manager import job_manager
from backend.schemas import TrainStartResponse, TrainStatusResponse

router = APIRouter(prefix="/api/train", tags=["training"])


def _run_training(job_id: str) -> None:
    """Background thread function that runs the full training pipeline."""
    job = job_manager.get_job(job_id)
    if job is None:
        return

    def callback(data: dict) -> None:
        job.update_from_callback(data)

    try:
        train_module.main(progress_callback=callback)
        job_manager.complete_job(job_id)
    except Exception as e:
        traceback.print_exc()
        job_manager.fail_job(job_id, str(e))


@router.post("/start", response_model=TrainStartResponse)
async def start_training() -> TrainStartResponse:
    """Kick off Stage 1 + Stage 2 training as a background task."""
    if job_manager.has_running_job("train"):
        raise HTTPException(
            status_code=409,
            detail="A training job is already running.",
        )

    job = job_manager.create_job("train")
    thread = threading.Thread(target=_run_training, args=(job.id,), daemon=True)
    thread.start()

    return TrainStartResponse(
        job_id=job.id,
        message="Training started in background.",
    )


@router.get("/status/{job_id}", response_model=TrainStatusResponse)
async def training_status(job_id: str) -> TrainStatusResponse:
    """Poll current training progress."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    return TrainStatusResponse(
        job_id=job.id,
        status=job.status,
        stage=job.stage,
        epoch=job.epoch,
        total_epochs=job.total_epochs,
        chunk=job.chunk,
        total_chunks=job.total_chunks,
        latest_loss=job.latest_loss,
        baseline_error=job.baseline_error,
        logs=job.logs[-100:],  # last 100 log lines
        error=job.error,
    )


@router.get("/stream/{job_id}")
async def training_stream(job_id: str) -> StreamingResponse:
    """SSE endpoint for live training log stream."""
    import asyncio
    import json

    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    async def event_generator():
        last_log_idx = 0
        while True:
            current_job = job_manager.get_job(job_id)
            if current_job is None:
                break

            # Send any new log entries
            new_logs = current_job.logs[last_log_idx:]
            if new_logs:
                last_log_idx = len(current_job.logs)
                data = json.dumps({
                    "status": current_job.status,
                    "stage": current_job.stage,
                    "epoch": current_job.epoch,
                    "total_epochs": current_job.total_epochs,
                    "chunk": current_job.chunk,
                    "total_chunks": current_job.total_chunks,
                    "latest_loss": current_job.latest_loss,
                    "baseline_error": current_job.baseline_error,
                    "new_logs": new_logs,
                    "error": current_job.error,
                })
                yield f"data: {data}\n\n"

            if current_job.status in ("completed", "failed"):
                # Send final state
                data = json.dumps({
                    "status": current_job.status,
                    "stage": current_job.stage,
                    "epoch": current_job.epoch,
                    "total_epochs": current_job.total_epochs,
                    "chunk": current_job.chunk,
                    "total_chunks": current_job.total_chunks,
                    "latest_loss": current_job.latest_loss,
                    "baseline_error": current_job.baseline_error,
                    "new_logs": [],
                    "error": current_job.error,
                })
                yield f"data: {data}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
