"""
Tool: get_current_machine_state
Extracts current Phase 1 anomaly magnitude, Phase 2D 48h failure risk, persistence, and state categorization.
"""
import os
import json
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.phase3.schemas.schemas import MachineStateOutput


def get_current_machine_state(
    record: Dict[str, Any],
    frozen_policy: Optional[Dict[str, Any]] = None,
) -> MachineStateOutput:
    """Deterministic tool computing machine state classification from verified pipeline outputs.
    
    Args:
        record: Dictionary containing:
            - timestamp: str
            - anomaly_score: float (Phase 1 MSE)
            - failure_probability_48h: float (Phase 2D Risk Prob)
            - persistence_windows: int
            - anomaly_slope: float
            - risk_slope: float
        frozen_policy: Dict loaded from checkpoints/phase2d/phase2d_frozen_policy.json
    """
    ts = str(record["timestamp"])
    score = float(record.get("anomaly_score", 0.0))
    prob = float(record.get("failure_probability_48h", 0.0))
    persist_w = int(record.get("persistence_windows", 0))
    a_slope = float(record.get("anomaly_slope", 0.0))
    r_slope = float(record.get("risk_slope", 0.0))

    p1_thresh = float(record.get("phase1_threshold", 2.913448))
    
    if frozen_policy is not None:
        p2_thresh = float(frozen_policy.get("frozen_policy", {}).get("threshold", 0.020))
        p2_persist = int(frozen_policy.get("frozen_policy", {}).get("persistence", 20))
    else:
        p2_thresh = 0.020
        p2_persist = 20

    # Determine Anomaly Severity
    if score <= p1_thresh:
        anom_severity = "NORMAL"
    elif score <= 1.5 * p1_thresh:
        anom_severity = "ELEVATED"
    elif score <= 3.0 * p1_thresh:
        anom_severity = "HIGH"
    else:
        anom_severity = "CRITICAL"

    # Determine Risk State
    if prob < p2_thresh:
        risk_state = "LOW"
    elif prob < 0.10:
        risk_state = "MODERATE"
    elif prob < 0.50:
        risk_state = "HIGH"
    else:
        risk_state = "CRITICAL"

    # Trends
    if a_slope > 0.005:
        anom_trend = "INCREASING"
    elif a_slope < -0.005:
        anom_trend = "DECREASING"
    else:
        anom_trend = "STABLE"

    if r_slope > 0.005:
        risk_trend = "INCREASING"
    elif r_slope < -0.005:
        risk_trend = "DECREASING"
    else:
        risk_trend = "STABLE"

    # Machine State Synthesis (State Machine)
    # Critical: severe persistent anomaly OR very high failure probability
    if (prob >= 0.50 and persist_w >= p2_persist) or (score > 3.0 * p1_thresh and persist_w >= 10):
        machine_state = "CRITICAL"
        requires_diag = True
    # High Risk: 48h risk threshold exceeded with sustained persistence OR high anomaly
    elif (prob >= p2_thresh and persist_w >= p2_persist) or (score > p1_thresh and persist_w >= 10):
        machine_state = "HIGH_RISK"
        requires_diag = True
    # Watch: anomaly elevated or risk moderately elevated without full persistence
    elif (score > p1_thresh) or (prob >= p2_thresh):
        machine_state = "WATCH"
        requires_diag = True
    # Normal: healthy operation
    else:
        machine_state = "NORMAL"
        requires_diag = False

    return MachineStateOutput(
        timestamp=ts,
        machine_state=machine_state,
        anomaly_severity=anom_severity,
        failure_risk_48h=round(prob, 4),
        risk_state=risk_state,
        anomaly_trend=anom_trend,
        risk_trend=risk_trend,
        requires_diagnostic_analysis=requires_diag,
    )
