"""
Phase 3 Schemas: Structured Data Models for Machine State, Diagnosis,
Historical Similarity, Risk Assessment, and Maintenance Decision Records.
"""
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass, field, asdict
import json


@dataclass
class MachineStateOutput:
    timestamp: str
    machine_state: Literal["NORMAL", "WATCH", "HIGH_RISK", "CRITICAL"]
    anomaly_severity: Literal["NORMAL", "ELEVATED", "HIGH", "CRITICAL"]
    failure_risk_48h: float
    risk_state: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    anomaly_trend: Literal["DECREASING", "STABLE", "INCREASING"]
    risk_trend: Literal["DECREASING", "STABLE", "INCREASING"]
    requires_diagnostic_analysis: bool


@dataclass
class SensorAttributionOutput:
    timestamp: str
    sensors: Dict[str, Dict[str, float]]  # {sensor_name: {"error": float, "contribution": float}}
    dominant_sensor: str
    dominant_contribution_pct: float
    dominant_sensor_stability_pct: float


@dataclass
class TrendAnalysisOutput:
    timestamp: str
    anomaly_slope_short: float   # ~30m
    anomaly_slope_medium: float  # ~2h
    anomaly_slope_long: float    # ~6h
    anomaly_acceleration: float
    persistence_windows: int
    persistence_minutes: float
    dominant_sensor_stability: float


@dataclass
class HistoricalMatchOutput:
    most_similar_failure: Optional[str]  # e.g., "Failure #3" or None
    similarity_score: float             # [0.0, 1.0]
    failure_type: Optional[str]         # e.g., "Air leak"
    failure_start: Optional[str]
    matching_indicators: List[str]
    differences: List[str]
    is_strong_match: bool


@dataclass
class DiagnosticOutput:
    primary_indicator: str
    supporting_indicators: List[str]
    pattern_type: str  # e.g., "DISCHARGE_PRESSURE_ANOMALY", "THERMAL_DRIFT", "ELECTROMECHANICAL_FLUCTUATION"
    evidence: List[str]
    confidence: float


@dataclass
class RiskReasoningOutput:
    risk_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    risk_interpretation: str
    warning_window: str  # "WITHIN_48_HOURS"
    confidence: float
    evidence: List[str]


@dataclass
class MaintenanceRecommendationOutput:
    action: Literal[
        "MONITOR",
        "INCREASE_MONITORING",
        "SCHEDULE_INSPECTION",
        "PRIORITY_INSPECTION",
        "IMMEDIATE_MAINTENANCE_REVIEW",
    ]
    urgency: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    recommended_area: str  # e.g., "pneumatic/discharge subsystem (DV_pressure, TP2, TP3)"
    reason: List[str]


@dataclass
class StructuredDecisionRecord:
    timestamp: str
    machine_state: Literal["NORMAL", "WATCH", "HIGH_RISK", "CRITICAL"]
    anomaly: Dict[str, Any]  # {"score": float, "severity": str, "persistent": bool}
    prediction: Dict[str, Any]  # {"horizon_hours": 48, "failure_probability": float, "risk_level": str}
    diagnosis: Dict[str, Any]  # {"primary_indicator": str, "supporting_indicators": [], "pattern_type": str, "confidence": float}
    historical_context: Dict[str, Any]  # {"similar_failure": str, "similarity": float, "matching_features": []}
    maintenance: Dict[str, Any]  # {"action": str, "urgency": str, "recommended_area": str}
    reasoning: List[str]
    evidence: List[str]
    agent_trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
