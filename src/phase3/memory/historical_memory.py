"""
Phase 3 Memory: HistoricalMemory
Provides immutable, verified profiles of MetroPT-3 compressor failures (#1, #2, #3, #4)
derived from sensor data, without inventing unverified repair logs or false details.
"""
from typing import Dict, List, Any, Optional
from src.phase3.tools.historical_search_tool import DOCUMENTED_FAILURE_PROFILES


class HistoricalMemory:
    """Read-only memory repository of documented MetroPT-3 failure events and precursor characteristics."""

    def __init__(self):
        self.profiles = {p["name"]: p for p in DOCUMENTED_FAILURE_PROFILES}

    def get_all_profiles(self) -> List[Dict[str, Any]]:
        return list(self.profiles.values())

    def get_profile(self, failure_name: str) -> Optional[Dict[str, Any]]:
        return self.profiles.get(failure_name)

    def describe_failure(self, failure_name: str) -> str:
        prof = self.get_profile(failure_name)
        if not prof:
            return f"No documented profile found for {failure_name}."
        return (
            f"{prof['name']} ({prof['failure_type']}, starting {prof['failure_start']}): "
            f"Dominant indicator {prof['dominant_sensor']} (contribution ~{prof['contributions'][prof['dominant_sensor']]*100:.1f}%). "
            f"Precursor characteristic: {prof['precursor_pattern']}."
        )
