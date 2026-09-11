"""
Tool: search_historical_failures
Performs deterministic similarity matching between current machine condition
and documented MetroPT-3 compressor failure precursor profiles (Failures #1, #2, #3, #4).
"""
from typing import Dict, List, Any, Optional
import numpy as np
from src.phase3.schemas.schemas import HistoricalMatchOutput

try:
    from src.config import ANALOGUE_SENSORS, KNOWN_FAILURES
except ImportError:
    from config import ANALOGUE_SENSORS, KNOWN_FAILURES


# Documented MetroPT-3 Failure Profiles (Precursor characteristics derived from dataset)
DOCUMENTED_FAILURE_PROFILES = [
    {
        "failure_id": 1,
        "name": "Failure #1",
        "failure_type": "Air leak",
        "failure_start": "2020-04-18 00:00:00",
        "dominant_sensor": "DV_pressure",
        "contributions": {
            "DV_pressure": 0.88,
            "H1": 0.04,
            "TP2": 0.03,
            "TP3": 0.02,
            "Reservoirs": 0.01,
            "Motor_current": 0.01,
            "Oil_temperature": 0.01,
        },
        "mean_precursor_mse": 3.42,
        "precursor_persistence_min": 180.0,
        "precursor_pattern": "Unimodal DV_pressure surge with low motor/thermal deviation",
    },
    {
        "failure_id": 2,
        "name": "Failure #2",
        "failure_type": "Air leak",
        "failure_start": "2020-05-29 23:30:00",
        "dominant_sensor": "DV_pressure",
        "contributions": {
            "DV_pressure": 0.76,
            "H1": 0.09,
            "TP2": 0.06,
            "TP3": 0.04,
            "Reservoirs": 0.02,
            "Motor_current": 0.02,
            "Oil_temperature": 0.01,
        },
        "mean_precursor_mse": 2.85,
        "precursor_persistence_min": 120.0,
        "precursor_pattern": "Discharge pressure leakage with moderate H1 pneumatics coupling",
    },
    {
        "failure_id": 3,
        "name": "Failure #3",
        "failure_type": "Air leak",
        "failure_start": "2020-06-05 10:00:00",
        "dominant_sensor": "DV_pressure",
        "contributions": {
            "DV_pressure": 0.92,
            "H1": 0.03,
            "TP2": 0.02,
            "TP3": 0.01,
            "Reservoirs": 0.01,
            "Motor_current": 0.005,
            "Oil_temperature": 0.005,
        },
        "mean_precursor_mse": 24.24,
        "precursor_persistence_min": 240.0,
        "precursor_pattern": "Severe accelerating DV_pressure anomaly with high persistence (>4h)",
    },
    {
        "failure_id": 4,
        "name": "Failure #4",
        "failure_type": "Air leak",
        "failure_start": "2020-07-15 14:30:00",
        "dominant_sensor": "DV_pressure",
        "contributions": {
            "DV_pressure": 0.55,
            "H1": 0.22,
            "TP2": 0.12,
            "TP3": 0.05,
            "Reservoirs": 0.03,
            "Motor_current": 0.02,
            "Oil_temperature": 0.01,
        },
        "mean_precursor_mse": 4.15,
        "precursor_persistence_min": 60.0,
        "precursor_pattern": "Multi-sensor air-leak precursor with elevated H1 and TP2 pneumatic cross-coupling",
    },
]


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes cosine similarity between two positive vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def search_historical_failures(
    current_contributions: Dict[str, float],
    current_anomaly_score: float,
    current_persistence_minutes: float,
    current_dominant_sensor: str,
) -> HistoricalMatchOutput:
    """Matches current sensor contributions and temporal stats against historical failure precursors.
    
    Similarity Metric = 0.70 * Cosine(Contributions) + 0.15 * DominantMatch + 0.15 * AnomalyProximity
    """
    cur_vec = np.array([current_contributions.get(s, 0.0) / 100.0 for s in ANALOGUE_SENSORS], dtype=np.float64)

    best_match = None
    best_score = -1.0
    best_profile = None

    for profile in DOCUMENTED_FAILURE_PROFILES:
        prof_vec = np.array([profile["contributions"].get(s, 0.0) for s in ANALOGUE_SENSORS], dtype=np.float64)
        cos_sim = cosine_similarity(cur_vec, prof_vec)

        # Dominant match bonus
        dom_match = 1.0 if current_dominant_sensor == profile["dominant_sensor"] else 0.0

        # Anomaly score proximity (log scale difference)
        log_diff = abs(np.log1p(current_anomaly_score) - np.log1p(profile["mean_precursor_mse"]))
        anom_sim = float(np.exp(-0.5 * log_diff))

        composite_sim = float(0.70 * cos_sim + 0.15 * dom_match + 0.15 * anom_sim)

        if composite_sim > best_score:
            best_score = composite_sim
            best_match = profile["name"]
            best_profile = profile

    # Threshold for declaring meaningful historical similarity (0.65)
    is_strong = bool(best_score >= 0.65)

    matching_indicators = []
    differences = []

    if best_profile:
        if current_dominant_sensor == best_profile["dominant_sensor"]:
            matching_indicators.append(f"{current_dominant_sensor} dominance matches {best_profile['name']}")
        else:
            differences.append(f"Current dominant is {current_dominant_sensor} vs {best_profile['dominant_sensor']} in {best_profile['name']}")

        if best_score >= 0.75:
            matching_indicators.append(f"Sensor attribution profile matches {best_profile['name']} (similarity {best_score*100:.1f}%)")
            matching_indicators.append(best_profile["precursor_pattern"])

        if current_persistence_minutes >= 30.0:
            matching_indicators.append(f"Persistent anomaly ({current_persistence_minutes:.0f} min) matches pre-failure duration pattern")
        else:
            differences.append("Current anomaly persistence is shorter than historical failure build-up")

    return HistoricalMatchOutput(
        most_similar_failure=best_match if is_strong else None,
        similarity_score=round(best_score, 4),
        failure_type=best_profile["failure_type"] if is_strong and best_profile else None,
        failure_start=best_profile["failure_start"] if is_strong and best_profile else None,
        matching_indicators=matching_indicators if is_strong else [],
        differences=differences,
        is_strong_match=is_strong,
    )
