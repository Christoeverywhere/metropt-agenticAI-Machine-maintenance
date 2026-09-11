"""
Phase 3 Evaluator: Mode A — Rule-Based Baseline
Executes fixed hardcoded heuristic decision rules without agent orchestration or LLM reasoning.
"""
from typing import Dict, List, Any
import time
from src.phase3.schemas.schemas import StructuredDecisionRecord


def run_rule_based_baseline(scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluates Mode A: Rule-Based Baseline on provided scenarios."""
    decisions = []

    for sc in scenarios:
        rec = sc["record"]
        ts = rec["timestamp"]
        score = float(rec["anomaly_score"])
        prob = float(rec["failure_probability_48h"])
        persist = int(rec.get("persistence_windows", 0))
        dominant = rec.get("dominant_sensor", "DV_pressure")

        start = time.perf_counter()

        # Fixed rule logic
        if prob >= 0.50 or score > 10.0:
            state = "CRITICAL"
            action = "IMMEDIATE_MAINTENANCE_REVIEW"
            urgency = "CRITICAL"
        elif prob >= 0.020 and persist >= 20:
            state = "HIGH_RISK"
            action = "PRIORITY_INSPECTION"
            urgency = "HIGH"
        elif score > 2.913448 or prob >= 0.020:
            state = "WATCH"
            action = "INCREASE_MONITORING"
            urgency = "MEDIUM"
        else:
            state = "NORMAL"
            action = "MONITOR"
            urgency = "LOW"

        lat = (time.perf_counter() - start) * 1000.0

        d = StructuredDecisionRecord(
            timestamp=ts,
            machine_state=state,
            anomaly={"score": score, "severity": "ELEVATED" if score > 2.91 else "NORMAL", "persistent": persist >= 10},
            prediction={"horizon_hours": 48, "failure_probability": prob, "risk_level": "HIGH" if prob >= 0.02 else "LOW"},
            diagnosis={"primary_indicator": dominant, "supporting_indicators": [], "pattern_type": "RULE_HEURISTIC", "confidence": 0.80},
            historical_context={"similar_failure": "None", "similarity": 0.0, "matching_features": []},
            maintenance={"action": action, "urgency": urgency, "recommended_area": f"{dominant} Subsystem"},
            reasoning=["Rule-based threshold condition triggered."],
            evidence=[f"Score: {score:.2f}", f"Prob: {prob:.4f}"],
            agent_trace={
                "mode": "RULE_BASED",
                "agents_called": ["rule_engine"],
                "tools_called": ["fixed_threshold_eval"],
                "tool_count": 1,
                "latency_ms": round(lat, 3),
            },
        )
        decisions.append(d.to_dict())

    return decisions
