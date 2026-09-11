"""
Tool: get_sensor_attribution
Decomposes reconstruction errors across the 7 analogue sensors, calculates contribution percentages,
and identifies dominant anomaly indicator and stability.
"""
from typing import Dict, List, Any, Optional
import numpy as np
from src.phase3.schemas.schemas import SensorAttributionOutput

try:
    from src.config import ANALOGUE_SENSORS
except ImportError:
    from config import ANALOGUE_SENSORS


def get_sensor_attribution(
    timestamp: str,
    sensor_errors_dict: Dict[str, float],
    recent_dominant_sensors: Optional[List[str]] = None,
) -> SensorAttributionOutput:
    """Calculates deterministic sensor-level error decomposition and contributions.
    
    Args:
        timestamp: str
        sensor_errors_dict: {sensor_name: mse_error}
        recent_dominant_sensors: Optional list of recent dominant sensor names to calculate stability
    """
    total_err = sum(max(0.0, float(v)) for v in sensor_errors_dict.values())
    denom = max(total_err, 1e-8)

    sensors_detail = {}
    contributions = {}

    for s_name in ANALOGUE_SENSORS:
        err = float(sensor_errors_dict.get(s_name, 0.0))
        pct = float((err / denom) * 100.0)
        sensors_detail[s_name] = {
            "error": round(err, 4),
            "contribution": round(pct, 2),
            "contribution_pct": round(pct, 2),
        }
        contributions[s_name] = pct

    # Dominant sensor (highest contribution)
    dominant_sensor = max(contributions.items(), key=lambda x: x[1])[0]
    dominant_contrib = contributions[dominant_sensor]

    # Calculate stability if recent history provided
    if recent_dominant_sensors and len(recent_dominant_sensors) > 0:
        match_count = sum(1 for s in recent_dominant_sensors if s == dominant_sensor)
        stability_pct = float((match_count / len(recent_dominant_sensors)) * 100.0)
    else:
        stability_pct = 100.0

    return SensorAttributionOutput(
        timestamp=str(timestamp),
        sensors=sensors_detail,
        dominant_sensor=dominant_sensor,
        dominant_contribution_pct=round(dominant_contrib, 2),
        dominant_sensor_stability_pct=round(stability_pct, 2),
    )
