"""
Phase 3 Agent: DiagnosticAgent
Explains which sensors contribute to abnormal conditions, distinguishes indicator from root cause,
and categorizes anomaly pattern types.
"""
from typing import Dict, List, Any
from src.phase3.schemas.schemas import DiagnosticOutput, SensorAttributionOutput


class DiagnosticAgent:
    """Agent that performs diagnostic interpretation of sensor reconstruction errors and attribution."""

    def diagnose(
        self,
        sensor_attrib: SensorAttributionOutput,
        anomaly_score: float,
        persistence_minutes: float,
    ) -> DiagnosticOutput:
        """Produces evidence-grounded diagnostic analysis without claiming unverified physical root causes."""
        dominant = sensor_attrib.dominant_sensor
        contrib = sensor_attrib.dominant_contribution_pct
        stability = sensor_attrib.dominant_sensor_stability_pct

        # Find supporting indicators (>10% contribution and not dominant)
        supporting = []
        for s_name, data in sensor_attrib.sensors.items():
            if s_name != dominant and data["contribution"] >= 10.0:
                supporting.append(s_name)

        # Classify pattern type
        if contrib >= 70.0 and dominant == "DV_pressure":
            pattern_type = "DISCHARGE_PRESSURE_DOMINANT_ANOMALY"
        elif "H1" in supporting or dominant == "H1":
            pattern_type = "INTAKE_OR_PNEUMATIC_CROSS_COUPLING"
        elif dominant in ("TP2", "TP3"):
            pattern_type = "INTERSTAGE_DELIVERY_PRESSURE_ANOMALY"
        elif dominant == "Motor_current":
            pattern_type = "ELECTROMECHANICAL_DRIVE_ANOMALY"
        elif dominant == "Oil_temperature":
            pattern_type = "THERMAL_MANAGEMENT_ANOMALY"
        elif len(supporting) >= 2:
            pattern_type = "MULTI_SENSOR_COUPLED_ANOMALY"
        else:
            pattern_type = "UNSPECIFIED_ANOMALOUS_CONDITION"

        # Formulate grounded evidence statements
        evidence = [
            f"{dominant} is the dominant anomaly indicator, accounting for {contrib:.1f}% of total reconstruction error",
        ]
        if supporting:
            evidence.append(
                f"Secondary supporting indicators with elevated errors include: {', '.join(supporting)}"
            )
        if persistence_minutes > 0:
            evidence.append(f"Abnormality has persisted continuously for {persistence_minutes:.1f} minutes")
        if stability >= 80.0:
            evidence.append(f"{dominant} indicator dominance has remained consistent ({stability:.1f}% stability)")

        # Calculate deterministic diagnostic confidence
        # Higher confidence when dominant contribution is high and indicator is stable over time
        conf = min(0.99, max(0.50, (contrib / 100.0) * 0.60 + (stability / 100.0) * 0.40))

        return DiagnosticOutput(
            primary_indicator=dominant,
            supporting_indicators=supporting,
            pattern_type=pattern_type,
            evidence=evidence,
            confidence=round(conf, 4),
        )
