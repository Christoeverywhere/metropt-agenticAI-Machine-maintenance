"""
Phase 3 Evaluator: Mode C — Agentic Predictive Maintenance System
Hierarchical multi-agent decision system featuring state-based dynamic routing,
specialized agent roles, short-term operational memory, and selective tool invocation.
"""
from typing import Dict, List, Any, Optional
from src.phase3.agents.supervisor import SupervisorAgent


def run_agentic_system(
    scenarios: List[Dict[str, Any]],
    frozen_policy_path: str = "checkpoints/phase2d/phase2d_frozen_policy.json",
) -> List[Dict[str, Any]]:
    """Evaluates Mode C: Proposed Agentic Multi-Agent System on scenarios."""
    supervisor = SupervisorAgent(frozen_policy_path=frozen_policy_path)
    decisions = []

    for sc in scenarios:
        rec = sc["record"]
        decision = supervisor.process_telemetry_window(rec, mode="AGENTIC")
        decisions.append(decision.to_dict())

    return decisions
