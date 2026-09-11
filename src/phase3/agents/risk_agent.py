"""
Phase 3 Agent: RiskReasoningAgent
Contextually interprets the Phase 2D 48-hour failure risk probability,
adhering strictly to probabilistic language without false certainty.
"""
from typing import Dict, List, Any, Optional
from src.phase3.schemas.schemas import RiskReasoningOutput, MachineStateOutput, TrendAnalysisOutput


class RiskReasoningAgent:
    """Agent that interprets failure probabilities in operational and temporal context."""

    def __init__(self, risk_threshold: float = 0.020):
        self.risk_threshold = risk_threshold

    def evaluate_risk(
        self,
        machine_state: MachineStateOutput,
        trend: TrendAnalysisOutput,
        historical_similarity: float = 0.0,
        similar_failure: Optional[str] = None,
    ) -> RiskReasoningOutput:
        """Evaluates predictive risk within the 48-hour operational window."""
        prob = machine_state.failure_risk_48h
        r_state = machine_state.risk_state
        r_trend = machine_state.risk_trend

        evidence = [
            f"Supervised 48h failure probability is {prob*100:.2f}% (operational entry threshold = {self.risk_threshold*100:.1f}%)",
            f"Risk trajectory is currently {r_trend.lower()}",
            f"Anomaly has persisted across {trend.persistence_windows} consecutive evaluation windows ({trend.persistence_minutes:.1f} min)",
        ]

        if prob >= 0.50:
            interpretation = (
                f"The predictive model indicates critical impending failure risk ({prob*100:.1f}%) within the 48-hour horizon, "
                f"characterized by severe sustained degradation."
            )
            conf = min(0.95, 0.70 + prob * 0.25)
        elif prob >= self.risk_threshold:
            interpretation = (
                f"The predictive model indicates an elevated probability ({prob*100:.1f}%) of entering a documented failure precursor "
                f"state within the 48-hour operational window."
            )
            conf = min(0.85, 0.60 + prob * 0.5)
        elif prob >= 0.010:
            interpretation = (
                f"The predictive model reflects minor risk fluctuation ({prob*100:.2f}%), below the operational escalation threshold "
                f"but within hysteresis exit bounds."
            )
            conf = 0.75
        else:
            interpretation = (
                f"Impending failure risk is negligible ({prob*100:.2f}%), indicating nominal operational margin over the next 48 hours."
            )
            conf = 0.90

        if similar_failure and historical_similarity >= 0.65:
            evidence.append(
                f"Risk profile correlates with {similar_failure} precursor signature (historical similarity {historical_similarity*100:.1f}%)"
            )

        return RiskReasoningOutput(
            risk_level=r_state,
            risk_interpretation=interpretation,
            warning_window="WITHIN_48_HOURS",
            confidence=round(conf, 4),
            evidence=evidence,
        )
