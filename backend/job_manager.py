"""
In-memory job manager for background training and evaluation tasks.

Each job has:
  - id (uuid)
  - status: "running" | "completed" | "failed"
  - kind: "train" | "evaluate"
  - progress metadata (stage, epoch, chunk, loss, logs, error)
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Job:
    id: str
    kind: str  # "train" or "evaluate"
    status: str = "running"  # "running" | "completed" | "failed"
    stage: Optional[str] = None
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    chunk: Optional[int] = None
    total_chunks: Optional[int] = None
    latest_loss: Optional[float] = None
    baseline_error: Optional[float] = None
    logs: list[str] = field(default_factory=list)
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None

    def add_log(self, message: str) -> None:
        self.logs.append(message)

    def update_from_callback(self, data: dict[str, Any]) -> None:
        """Update job state from a training/evaluation progress callback."""
        if "stage" in data:
            self.stage = data["stage"]
        if "epoch" in data:
            self.epoch = data["epoch"]
        if "total_epochs" in data:
            self.total_epochs = data["total_epochs"]
        if "epochs" in data:
            self.total_epochs = data["epochs"]
        if "loss" in data:
            self.latest_loss = data["loss"]
        if "train_mse" in data:
            self.latest_loss = data["train_mse"]
        if "chunk" in data:
            self.chunk = data["chunk"]
        if "total_chunks" in data:
            self.total_chunks = data["total_chunks"]
        if "chunk_error" in data:
            self.latest_loss = data["chunk_error"]
        if "mean_error" in data:
            self.latest_loss = data["mean_error"]
        if "baseline_error" in data:
            self.baseline_error = data["baseline_error"]
        if "message" in data:
            self.add_log(data["message"])


class JobManager:
    """Thread-safe in-memory job tracker."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(self, kind: str) -> Job:
        job_id = str(uuid.uuid4())[:8]
        job = Job(id=job_id, kind=kind)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def complete_job(self, job_id: str, result: dict[str, Any] | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "completed"
                job.result = result

    def fail_job(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "failed"
                job.error = error

    def has_running_job(self, kind: str) -> bool:
        with self._lock:
            return any(
                j.kind == kind and j.status == "running"
                for j in self._jobs.values()
            )


# Singleton instance
job_manager = JobManager()
