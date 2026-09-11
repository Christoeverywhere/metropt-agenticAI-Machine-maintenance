"""
Pydantic models for all API request and response types.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DatasetUploadResponse(BaseModel):
    status: str
    row_count: int
    date_start: str
    date_end: str
    columns: list[str]


class SensorMissing(BaseModel):
    sensor: str
    missing_count: int
    missing_pct: float


class DatasetSummaryResponse(BaseModel):
    row_count: int
    date_start: str
    date_end: str
    healthy_train_rows: int
    stream_rows: int
    missing_values: list[SensorMissing]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TrainStartResponse(BaseModel):
    job_id: str
    message: str


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str  # "running" | "completed" | "failed" | "not_found"
    stage: Optional[str] = None
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    chunk: Optional[int] = None
    total_chunks: Optional[int] = None
    latest_loss: Optional[float] = None
    baseline_error: Optional[float] = None
    logs: list[str] = []
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvalStartResponse(BaseModel):
    job_id: str
    message: str


class EvalStatusResponse(BaseModel):
    job_id: str
    status: str
    logs: list[str] = []
    error: Optional[str] = None


class SensorContribution(BaseModel):
    sensor: str
    error: float


class FailureExplanation(BaseModel):
    failure_id: int
    failure_start: str
    top_sensors: list[SensorContribution]


class LeadTimeResult(BaseModel):
    failure_id: int
    failure_start: str
    detected: bool
    first_flag: Optional[str] = None
    lead_time_minutes: Optional[float] = None


class EvalResultsResponse(BaseModel):
    threshold: float
    mu: float
    sigma: float
    baseline_error: float
    lead_times: list[LeadTimeResult]
    explanations: list[FailureExplanation]
    n_scored_rows: int
    n_flagged: int


# ---------------------------------------------------------------------------
# Scores / sensor data
# ---------------------------------------------------------------------------


class ScorePoint(BaseModel):
    timestamp: str
    seq_error: float
    rolling_score: Optional[float] = None
    TP2_error: Optional[float] = None
    TP3_error: Optional[float] = None
    H1_error: Optional[float] = None
    DV_pressure_error: Optional[float] = None
    Reservoirs_error: Optional[float] = None
    Oil_temperature_error: Optional[float] = None
    Motor_current_error: Optional[float] = None


class ScoresResponse(BaseModel):
    points: list[dict]
    total_rows: int
    returned_rows: int
    start: str
    end: str


class SensorDataPoint(BaseModel):
    timestamp: str
    value: float


class SensorDataResponse(BaseModel):
    sensor: str
    scaled: bool
    points: list[dict]
    total_rows: int
    returned_rows: int


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


class FailureWindow(BaseModel):
    id: int
    start: str
    end: str
    type: str
    severity: str


class FailuresResponse(BaseModel):
    failures: list[FailureWindow]


# ---------------------------------------------------------------------------
# Model status
# ---------------------------------------------------------------------------


class ModelStatusResponse(BaseModel):
    exists: bool
    last_trained: Optional[str] = None
    baseline_error: Optional[float] = None
    scaler_exists: bool
    scored_stream_exists: bool
