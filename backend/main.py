"""
FastAPI backend for the MetroPT-3 Predictive Maintenance pipeline.

Wraps the src/ ML pipeline so a frontend can drive dataset management,
training, evaluation, and anomaly exploration via REST/SSE endpoints.

Run:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure src/ and project root are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas import HealthResponse
from backend.routes import dataset, training, evaluation, scores, model_status

app = FastAPI(
    title="MetroPT-3 Predictive Maintenance API",
    description="Backend service for LSTM Autoencoder anomaly detection pipeline",
    version="1.0.0",
)

# CORS — allow the frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all route modules
app.include_router(dataset.router)
app.include_router(training.router)
app.include_router(evaluation.router)
app.include_router(scores.router)
app.include_router(model_status.router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Basic liveness check."""
    return HealthResponse(status="ok", message="MetroPT-3 API is running.")
