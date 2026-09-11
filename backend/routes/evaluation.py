"""
Evaluation routes — run evaluation, get results.
"""

from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path

from fastapi import APIRouter, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
import evaluate as eval_module

from backend.job_manager import job_manager
from backend.schemas import (
    EvalStartResponse,
    EvalStatusResponse,
    EvalResultsResponse,
    LeadTimeResult,
    FailureExplanation,
    SensorContribution,
)

router = APIRouter(prefix="/api/evaluate", tags=["evaluation"])


def _run_evaluation(job_id: str) -> None:
    """Background thread function that runs the full evaluation pipeline."""
    job = job_manager.get_job(job_id)
    if job is None:
        return

    def callback(data: dict) -> None:
        job.update_from_callback(data)

    try:
        # eval_module.main() returns (threshold_info, scored, lead_time_results)
        threshold_info, scored, lead_time_results = eval_module.main()

        # Build structured results for the API
        explanations = []
        for r in lead_time_results:
            if r["detected"]:
                top_sensors = eval_module.top_contributing_sensors(scored, r["first_flag"])
                explanations.append({
                    "failure_id": r["failure_id"],
                    "failure_start": str(r["failure_start"]),
                    "top_sensors": [
                        {"sensor": col.replace("_error", ""), "error": val}
                        for col, val in top_sensors
                    ],
                })

        # Serialize lead_time_results for JSON
        serialized_lead_times = []
        for r in lead_time_results:
            serialized_lead_times.append({
                "failure_id": r["failure_id"],
                "failure_start": str(r["failure_start"]),
                "detected": r["detected"],
                "first_flag": str(r["first_flag"]) if r["first_flag"] is not None else None,
                "lead_time_minutes": r["lead_time_minutes"],
            })

        results = {
            "threshold": threshold_info["threshold"],
            "mu": threshold_info["mean"],
            "sigma": threshold_info["std"],
            "baseline_error": eval_module.load_model()[1],  # baseline_error from checkpoint
            "lead_times": serialized_lead_times,
            "explanations": explanations,
            "n_scored_rows": len(scored),
            "n_flagged": int((scored["seq_error"] > threshold_info["threshold"]).sum()),
        }
        job_manager.complete_job(job_id, result=results)
    except Exception as e:
        traceback.print_exc()
        job_manager.fail_job(job_id, str(e))


@router.post("/run", response_model=EvalStartResponse)
async def run_evaluation() -> EvalStartResponse:
    """Kick off threshold calibration + dense scoring + lead-time evaluation."""
    if not Path(config.MODEL_PATH).exists():
        raise HTTPException(
            status_code=404,
            detail="No trained model found. Run training first.",
        )

    if job_manager.has_running_job("evaluate"):
        raise HTTPException(
            status_code=409,
            detail="An evaluation job is already running.",
        )

    job = job_manager.create_job("evaluate")
    thread = threading.Thread(target=_run_evaluation, args=(job.id,), daemon=True)
    thread.start()

    return EvalStartResponse(
        job_id=job.id,
        message="Evaluation started in background.",
    )


@router.get("/status/{job_id}", response_model=EvalStatusResponse)
async def evaluation_status(job_id: str) -> EvalStatusResponse:
    """Poll current evaluation progress."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    return EvalStatusResponse(
        job_id=job.id,
        status=job.status,
        logs=job.logs[-100:],
        error=job.error,
    )


@router.get("/results", response_model=EvalResultsResponse)
async def evaluation_results() -> EvalResultsResponse:
    """Return structured evaluation results from the last completed evaluation."""
    # Find the last completed evaluation job
    result = None
    for job in reversed(list(job_manager._jobs.values())):
        if job.kind == "evaluate" and job.status == "completed" and job.result:
            result = job.result
            break

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No completed evaluation found. Run evaluation first.",
        )

    lead_times = [LeadTimeResult(**lt) for lt in result["lead_times"]]
    explanations = [
        FailureExplanation(
            failure_id=exp["failure_id"],
            failure_start=exp["failure_start"],
            top_sensors=[
                SensorContribution(**sc) for sc in exp["top_sensors"]
            ],
        )
        for exp in result["explanations"]
    ]

    return EvalResultsResponse(
        threshold=result["threshold"],
        mu=result["mu"],
        sigma=result["sigma"],
        baseline_error=result["baseline_error"],
        lead_times=lead_times,
        explanations=explanations,
        n_scored_rows=result["n_scored_rows"],
        n_flagged=result["n_flagged"],
    )
