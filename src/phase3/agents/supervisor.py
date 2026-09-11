"""
Phase 3 Agent: SupervisorAgent
Main orchestrator that coordinates selective agent execution via state-based routing,
gathers evidence from deterministic tools, and synthesizes structured final decision records.
"""
import os
import time
import json
from typing import Dict, List, Any, Optional
import pandas as pd

from src.phase3.schemas.schemas import (
    MachineStateOutput,
    SensorAttributionOutput,
    TrendAnalysisOutput,
    HistoricalMatchOutput,
    DiagnosticOutput,
    RiskReasoningOutput,
    MaintenanceRecommendationOutput,
    StructuredDecisionRecord,
)
from src.phase3.tools.machine_state_tool import get_current_machine_state
from src.phase3.tools.sensor_attribution_tool import get_sensor_attribution
from src.phase3.tools.risk_prediction_tool import get_risk_prediction
from src.phase3.tools.trend_analysis_tool import get_recent_trend
from src.phase3.tools.historical_search_tool import search_historical_failures
from src.phase3.tools.maintenance_context_tool import generate_maintenance_context
from src.phase3.memory.operational_memory import OperationalMemory
from src.phase3.memory.historical_memory import HistoricalMemory

from src.phase3.agents.monitoring_agent import MonitoringAgent
from src.phase3.agents.diagnostic_agent import DiagnosticAgent
from src.phase3.agents.historical_agent import HistoricalSimilarityAgent
from src.phase3.agents.risk_agent import RiskReasoningAgent
from src.phase3.agents.maintenance_agent import MaintenanceReasoningAgent


class SupervisorAgent:
    """Orchestrator for the Agentic Predictive Maintenance Decision System."""

    def __init__(
        self,
        frozen_policy_path: str = "checkpoints/phase2d/phase2d_frozen_policy.json",
        operational_memory: Optional[OperationalMemory] = None,
        historical_memory: Optional[HistoricalMemory] = None,
    ):
        # Load frozen policy
        self.frozen_policy = None
        if os.path.exists(frozen_policy_path):
            with open(frozen_policy_path, "r") as f:
                self.frozen_policy = json.load(f)
        
        self.risk_threshold = (
            self.frozen_policy.get("frozen_policy", {}).get("threshold", 0.020)
            if self.frozen_policy
            else 0.020
        )

        # Initialize sub-agents
        self.monitoring_agent = MonitoringAgent(frozen_policy=self.frozen_policy)
        self.diagnostic_agent = DiagnosticAgent()
        self.historical_agent = HistoricalSimilarityAgent()
        self.risk_agent = RiskReasoningAgent(risk_threshold=self.risk_threshold)
        self.maintenance_agent = MaintenanceReasoningAgent()

        # Initialize memory
        self.op_memory = operational_memory or OperationalMemory()
        self.hist_memory = historical_memory or HistoricalMemory()

    def process_telemetry_window(
        self,
        record: Dict[str, Any],
        history_df: Optional[pd.DataFrame] = None,
        mode: str = "AGENTIC",
    ) -> StructuredDecisionRecord:
        """Processes a single telemetry record and generates an evidence-grounded decision.
        
        Modes supported:
            - 'AGENTIC': State-based routing with selective agent/tool invocation.
            - 'SINGLE_LLM': Monolithic all-tool synthesis.
            - 'RULE_BASED': Fixed deterministic heuristic without agent decomposition.
        """
        start_time = time.perf_counter()
        tools_called = []
        agents_called = []

        # 1. Monitoring / State Assessment (Always called)
        tools_called.append("get_current_machine_state")
        m_state: MachineStateOutput = self.monitoring_agent.assess_state(record)
        agents_called.append("monitoring_agent")

        ts = m_state.timestamp
        state_str = m_state.machine_state

        # Update operational memory
        mem_update = self.op_memory.update_state(
            timestamp=ts,
            machine_state=state_str,
            anomaly_score=float(record.get("anomaly_score", 0.0)),
            failure_prob=m_state.failure_risk_48h,
            dominant_sensor=str(record.get("dominant_sensor", "DV_pressure")),
        )

        # 2. State-Based Routing (Selective Invocation)
        diagnostic: Optional[DiagnosticOutput] = None
        historical: Optional[HistoricalMatchOutput] = None
        risk: Optional[RiskReasoningOutput] = None
        maintenance: Optional[MaintenanceRecommendationOutput] = None
        trend: Optional[TrendAnalysisOutput] = None
        sensor_attrib: Optional[SensorAttributionOutput] = None

        # Helper to invoke sensor attribution and trend
        def _invoke_attribution_and_trend():
            s_errors = record.get("sensor_errors", {})
            if not s_errors:
                # Fallback default if record lacks sensor_errors dict
                dom = str(record.get("dominant_sensor", "DV_pressure"))
                sc = float(record.get("anomaly_score", 1.0))
                s_errors = {
                    "DV_pressure": sc * 0.70 if dom == "DV_pressure" else sc * 0.10,
                    "H1": sc * 0.70 if dom == "H1" else sc * 0.10,
                    "TP2": sc * 0.70 if dom == "TP2" else sc * 0.10,
                    "TP3": sc * 0.05,
                    "Reservoirs": sc * 0.02,
                    "Motor_current": sc * 0.02,
                    "Oil_temperature": sc * 0.01,
                }
            recent_states = self.op_memory.get_recent_states(count=20)
            recent_doms = [s["dominant_sensor"] for s in recent_states]
            recent_scores = [s["anomaly_score"] for s in recent_states] or [float(record.get("anomaly_score", 0.0))]
            persist_w = int(record.get("persistence_windows", 0))

            s_attrib = get_sensor_attribution(
                timestamp=ts,
                sensor_errors_dict=s_errors,
                recent_dominant_sensors=recent_doms,
            )
            tr = get_recent_trend(
                timestamp=ts,
                recent_scores=recent_scores,
                recent_dominant_sensors=recent_doms,
                persistence_windows=persist_w,
            )
            return s_attrib, tr

        if mode == "AGENTIC":
            if state_str == "NORMAL":
                # State A - Normal: Minimum overhead, no downstream agents needed
                diagnostic = None
                historical = None
                risk = None
                maintenance = self.maintenance_agent.recommend(m_state)
                agents_called.append("maintenance_agent")

            elif state_str == "WATCH":
                # State B - Watch: Call Diagnostic, selectively Trend
                tools_called.extend(["get_sensor_attribution", "get_recent_trend"])
                sensor_attrib, trend = _invoke_attribution_and_trend()

                diagnostic = self.diagnostic_agent.diagnose(
                    sensor_attrib=sensor_attrib,
                    anomaly_score=float(record.get("anomaly_score", 0.0)),
                    persistence_minutes=trend.persistence_minutes,
                )
                agents_called.append("diagnostic_agent")

                # If anomaly is moderately high, also check historical similarity
                if m_state.anomaly_severity in ("HIGH", "CRITICAL") or m_state.failure_risk_48h >= self.risk_threshold:
                    tools_called.append("search_historical_failures")
                    historical = self.historical_agent.compare_history(
                        sensor_attrib=sensor_attrib,
                        anomaly_score=float(record.get("anomaly_score", 0.0)),
                        persistence_minutes=trend.persistence_minutes,
                    )
                    agents_called.append("historical_similarity_agent")

                maintenance = self.maintenance_agent.recommend(
                    machine_state=m_state,
                    diagnostic=diagnostic,
                    historical=historical,
                    risk=risk,
                )
                agents_called.append("maintenance_agent")

            elif state_str in ("HIGH_RISK", "CRITICAL"):
                # States C & D - Escalation: Call full diagnostic, historical, risk, and maintenance agents
                tools_called.extend([
                    "get_sensor_attribution",
                    "get_recent_trend",
                    "search_historical_failures",
                    "generate_maintenance_context",
                ])
                sensor_attrib, trend = _invoke_attribution_and_trend()

                diagnostic = self.diagnostic_agent.diagnose(
                    sensor_attrib=sensor_attrib,
                    anomaly_score=float(record.get("anomaly_score", 0.0)),
                    persistence_minutes=trend.persistence_minutes,
                )
                agents_called.append("diagnostic_agent")

                historical = self.historical_agent.compare_history(
                    sensor_attrib=sensor_attrib,
                    anomaly_score=float(record.get("anomaly_score", 0.0)),
                    persistence_minutes=trend.persistence_minutes,
                )
                agents_called.append("historical_similarity_agent")

                risk = self.risk_agent.evaluate_risk(
                    machine_state=m_state,
                    trend=trend,
                    historical_similarity=historical.similarity_score if historical else 0.0,
                    similar_failure=historical.most_similar_failure if historical else None,
                )
                agents_called.append("risk_reasoning_agent")

                maintenance = self.maintenance_agent.recommend(
                    machine_state=m_state,
                    diagnostic=diagnostic,
                    historical=historical,
                    risk=risk,
                )
                agents_called.append("maintenance_agent")

        elif mode in ("SINGLE_LLM", "RULE_BASED"):
            # Unconditionally evaluate all components
            tools_called.extend([
                "get_sensor_attribution",
                "get_recent_trend",
                "search_historical_failures",
                "generate_maintenance_context",
            ])
            sensor_attrib, trend = _invoke_attribution_and_trend()
            diagnostic = self.diagnostic_agent.diagnose(
                sensor_attrib=sensor_attrib,
                anomaly_score=float(record.get("anomaly_score", 0.0)),
                persistence_minutes=trend.persistence_minutes,
            )
            historical = self.historical_agent.compare_history(
                sensor_attrib=sensor_attrib,
                anomaly_score=float(record.get("anomaly_score", 0.0)),
                persistence_minutes=trend.persistence_minutes,
            )
            risk = self.risk_agent.evaluate_risk(
                machine_state=m_state,
                trend=trend,
                historical_similarity=historical.similarity_score if historical else 0.0,
                similar_failure=historical.most_similar_failure if historical else None,
            )
            maintenance = self.maintenance_agent.recommend(
                machine_state=m_state,
                diagnostic=diagnostic,
                historical=historical,
                risk=risk,
            )
            agents_called.extend([
                "diagnostic_agent",
                "historical_similarity_agent",
                "risk_reasoning_agent",
                "maintenance_agent",
            ])

        # 3. Assemble Structured Evidence & Synthesis
        evidence_list = []
        evidence_list.append(f"Phase 1 Anomaly Score: {float(record.get('anomaly_score', 0.0)):.4f} (Severity: {m_state.anomaly_severity})")
        evidence_list.append(f"Phase 2D 48h Failure Risk: {m_state.failure_risk_48h*100:.2f}% (Risk State: {m_state.risk_state})")

        if diagnostic:
            evidence_list.append(f"Dominant Indicator: {diagnostic.primary_indicator} ({diagnostic.pattern_type})")
            if diagnostic.supporting_indicators:
                evidence_list.append(f"Supporting Indicators: {', '.join(diagnostic.supporting_indicators)}")
        
        if trend and trend.persistence_minutes > 0:
            evidence_list.append(f"Temporal Persistence: {trend.persistence_minutes:.1f} minutes")

        if historical and historical.is_strong_match:
            evidence_list.append(f"Historical Analogy: Strong match to {historical.most_similar_failure} ({historical.similarity_score*100:.1f}% similarity)")

        # Reasoning Trace
        reasoning_list = []
        if state_str == "NORMAL":
            reasoning_list.append("Reconstruction errors and 48-hour failure risk are within nominal operational bounds.")
            reasoning_list.append("Routine telemetry monitoring is maintained without unnecessary maintenance escalation.")
        elif state_str == "WATCH":
            reasoning_list.append("Machine telemetry exhibits mild statistical deviation from nominal operating baseline.")
            if diagnostic:
                reasoning_list.append(f"Localized anomaly observed primarily on {diagnostic.primary_indicator}.")
            reasoning_list.append("Increased telemetry monitoring or opportunistic depot inspection is recommended.")
        elif state_str == "HIGH_RISK":
            reasoning_list.append("Supervised 48-hour impending failure risk has breached operational policy with confirmed persistence.")
            if diagnostic:
                reasoning_list.append(f"Reconstruction error is heavily concentrated in {diagnostic.primary_indicator}.")
            if historical and historical.is_strong_match:
                reasoning_list.append(f"Degradation signature aligns with historical {historical.most_similar_failure} air-leak precursor.")
            reasoning_list.append("Priority maintenance inspection is required to prevent unscheduled downtime.")
        elif state_str == "CRITICAL":
            reasoning_list.append("Critical failure risk (>50%) or severe accelerating anomaly is actively developing.")
            reasoning_list.append("Immediate physical maintenance review and operational safety isolation are required.")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        decision_record = StructuredDecisionRecord(
            timestamp=ts,
            machine_state=m_state.machine_state,
            anomaly={
                "score": round(float(record.get("anomaly_score", 0.0)), 4),
                "severity": m_state.anomaly_severity,
                "persistent": bool(trend.persistence_windows >= 10) if trend else False,
            },
            prediction={
                "horizon_hours": 48,
                "failure_probability": m_state.failure_risk_48h,
                "risk_level": m_state.risk_state,
            },
            diagnosis={
                "primary_indicator": diagnostic.primary_indicator if diagnostic else "None",
                "supporting_indicators": diagnostic.supporting_indicators if diagnostic else [],
                "pattern_type": diagnostic.pattern_type if diagnostic else "NOMINAL_OPERATION",
                "confidence": diagnostic.confidence if diagnostic else 1.0,
            },
            historical_context={
                "similar_failure": historical.most_similar_failure if historical else "None",
                "similarity": historical.similarity_score if historical else 0.0,
                "matching_features": historical.matching_indicators if historical else [],
            },
            maintenance={
                "action": maintenance.action if maintenance else "MONITOR",
                "urgency": maintenance.urgency if maintenance else "LOW",
                "recommended_area": maintenance.recommended_area if maintenance else "General",
            },
            reasoning=reasoning_list,
            evidence=evidence_list,
            agent_trace={
                "mode": mode,
                "agents_called": agents_called,
                "tools_called": tools_called,
                "tool_count": len(tools_called),
                "latency_ms": round(elapsed_ms, 2),
                "alert_state": mem_update["current_alert_state"],
                "in_cooldown": mem_update["in_cooldown"],
            },
        )

        # Record decision in memory
        self.op_memory.record_decision(decision_record.to_dict())

        return decision_record
