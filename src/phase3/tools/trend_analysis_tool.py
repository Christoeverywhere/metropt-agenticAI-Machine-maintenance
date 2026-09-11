"""
Tool: get_recent_trend
Calculates causal trend slopes, rate-of-change acceleration, active persistence duration,
and indicator stability across multi-scale historical windows.
"""
from typing import Dict, List, Any, Optional
import numpy as np
from src.phase3.schemas.schemas import TrendAnalysisOutput


def compute_linear_slope(y_vals: np.ndarray) -> float:
    """Computes ordinary least squares slope over integer steps."""
    n = len(y_vals)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x_mean = (n - 1) / 2.0
    y_mean = np.mean(y_vals)
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    nom = np.sum((x - x_mean) * (y_vals - y_mean))
    return float(nom / denom)


def get_recent_trend(
    timestamp: str,
    recent_scores: List[float],
    recent_dominant_sensors: Optional[List[str]] = None,
    persistence_windows: int = 0,
    stride_seconds: float = 100.0,
) -> TrendAnalysisOutput:
    """Calculates multi-scale causal slopes and acceleration from recent anomaly score buffer.
    
    Args:
        timestamp: str
        recent_scores: List of chronological anomaly scores up to timestamp (length >= 1)
        recent_dominant_sensors: List of recent dominant sensor names
        persistence_windows: Count of consecutive anomalous windows
        stride_seconds: Seconds elapsed per window step (default: 100s for stride 10)
    """
    scores_arr = np.asarray(recent_scores, dtype=np.float64)
    n = len(scores_arr)

    # Multi-scale windows: short (6 ~30m), med (24 ~2h), long (72 ~6h)
    short_slice = scores_arr[-min(n, 6) :]
    med_slice = scores_arr[-min(n, 24) :]
    long_slice = scores_arr[-min(n, 72) :]

    slope_short = compute_linear_slope(short_slice)
    slope_med = compute_linear_slope(med_slice)
    slope_long = compute_linear_slope(long_slice)
    accel = slope_short - slope_med

    persist_min = (persistence_windows * stride_seconds) / 60.0

    if recent_dominant_sensors and len(recent_dominant_sensors) > 0:
        cur_dom = recent_dominant_sensors[-1]
        stab = float((sum(1 for s in recent_dominant_sensors if s == cur_dom) / len(recent_dominant_sensors)) * 100.0)
    else:
        stab = 100.0

    return TrendAnalysisOutput(
        timestamp=str(timestamp),
        anomaly_slope_short=round(slope_short, 6),
        anomaly_slope_medium=round(slope_med, 6),
        anomaly_slope_long=round(slope_long, 6),
        anomaly_acceleration=round(accel, 6),
        persistence_windows=int(persistence_windows),
        persistence_minutes=round(persist_min, 1),
        dominant_sensor_stability=round(stab, 2),
    )
