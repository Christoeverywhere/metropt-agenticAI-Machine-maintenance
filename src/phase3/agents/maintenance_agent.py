"""
Phase 3 Agent: MaintenanceReasoningAgent
Translates diagnostic, historical, and risk evidence into actionable operational recommendations.
Frames actions as inspection, review, and monitoring rather than unsupported replacement claims.
"""
from typing import Dict, List, Any, Optional
from src.phase3.schemas.schemas import (
    MaintenanceRecommendationOutput,
    MachineStateOutput,
    DiagnosticOutput,
    HistoricalMatchOutput,
    RiskReasoningOutput,
)
from src.phase3.tools.maintenance_context_tool import generate_maintenance_context


class MaintenanceReasoningAgent:
    """Agent that synthesizes multi-source evidence into structured maintenance actions."""

    def recommend(
        self,
        machine_state: MachineStateOutput,
        diagnostic: Optional[DiagnosticOutput] = None,
        historical: Optional[HistoricalMatchOutput] = None,
        risk: Optional[RiskReasoningOutput] = None,
    ) -> MaintenanceRecommendationOutput:
        """Determines operational maintenance actions grounded strictly in evidence."""
        m_state = machine_state.machine_state
        reasons = []

        # Subsystem resolution
        dom_sensor = diagnostic.primary_indicator if diagnostic else "DV_pressure"
        supp_sensors = diagnostic.supporting_indicators if diagnostic else []
        maint_ctx = generate_maintenance_context(
            dominant_sensor=dom_sensor,
            supporting_sensors=supp_sensors,
            machine_state=m_state,
        )

        area_desc = f"{maint_ctx['primary_subsystem']} ({dom_sensor}"
        if supp_sensors:
            area_desc += f", {', '.join(supp_sensors)}"
        area_desc += ")"

        if m_state == "CRITICAL":
            action = "IMMEDIATE_MAINTENANCE_REVIEW"
            urgency = "CRITICAL"
            reasons.append("Critical impending failure risk exceeds 50% or severe persistent anomaly is active")
            reasons.append(f"{dom_sensor} is the primary anomaly indicator in the {maint_ctx['primary_subsystem']}")
            if historical and historical.is_strong_match:
                reasons.append(f"Condition exhibits strong similarity ({historical.similarity_score*100:.1f}%) to historical {historical.most_similar_failure}")
            reasons.append("Immediate physical inspection and subsystem isolation review required before next operational cycle")

        elif m_state == "HIGH_RISK":
            action = "PRIORITY_INSPECTION"
            urgency = "HIGH"
            reasons.append("Supervised 48-hour failure risk policy threshold exceeded with sustained persistence")
            reasons.append(f"Reconstruction error concentrated in {dom_sensor} ({maint_ctx['primary_subsystem']})")
            if historical and historical.is_strong_match:
                reasons.append(f"Precursor profile resembles {historical.most_similar_failure} ({historical.similarity_score*100:.1f}% similarity)")
            reasons.append("Dispatch maintenance technician to inspect targeted subsystem components within 12-24 hours")

        elif m_state == "WATCH":
            # Primary action for WATCH state is INCREASE_MONITORING
            action = "INCREASE_MONITORING"
            urgency = "MEDIUM"
            reasons.append("Machine condition exhibits transient deviation from baseline nominal distribution")
            reasons.append(f"Diagnostic indicator localized to {dom_sensor}")
            reasons.append("Elevate telemetry sampling and track anomaly persistence over the next 6 hours")

        else:  # NORMAL
            action = "MONITOR"
            urgency = "LOW"
            area_desc = "All compressor subsystems (Nominal Baseline)"
            reasons.append("Sensor reconstruction errors and 48-hour impending failure risk remain within healthy operational bounds")
            reasons.append("Maintain standard predictive telemetry monitoring schedule")

        return MaintenanceRecommendationOutput(
            action=action,
            urgency=urgency,
            recommended_area=area_desc,
            reason=reasons,
        )
