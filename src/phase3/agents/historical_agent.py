"""
Phase 3 Agent: HistoricalSimilarityAgent
Compares current machine state with documented historical failure precursor patterns (Failures #1, #2, #3, #4).
"""
from typing import Dict, List, Any
from src.phase3.schemas.schemas import HistoricalMatchOutput, SensorAttributionOutput
from src.phase3.tools.historical_search_tool import search_historical_failures


class HistoricalSimilarityAgent:
    """Agent that performs pattern matching against documented MetroPT-3 compressor failures."""

    def compare_history(
        self,
        sensor_attrib: SensorAttributionOutput,
        anomaly_score: float,
        persistence_minutes: float,
    ) -> HistoricalMatchOutput:
        """Determines if the current sensor condition closely resembles a documented historical precursor."""
        # Convert sensor dict to contribution dict
        contrib_dict = {
            s: data["contribution"] for s, data in sensor_attrib.sensors.items()
        }

        return search_historical_failures(
            current_contributions=contrib_dict,
            current_anomaly_score=anomaly_score,
            current_persistence_minutes=persistence_minutes,
            current_dominant_sensor=sensor_attrib.dominant_sensor,
        )
