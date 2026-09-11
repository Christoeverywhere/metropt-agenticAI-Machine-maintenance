"""
Phase 3 Evaluator: Mode B — Single LLM Decision Maker
Monolithic decision maker that ingests all tool outputs simultaneously without agent decomposition or state routing.
"""
from typing import Dict, List, Any, Optional
from src.phase3.agents.supervisor import SupervisorAgent


def run_single_llm_baseline(
    scenarios: List[Dict[str, Any]],
    frozen_policy_path: str = "checkpoints/phase2d/phase2d_frozen_policy.json",
) -> List[Dict[str, Any]]:
    """Evaluates Mode B: Single monolithic decision maker on scenarios."""
    supervisor = SupervisorAgent(frozen_policy_path=frozen_policy_path)
    decisions = []

    for sc in scenarios:
        rec = sc["record"]
        # In Mode B, all tools and reasoning are unconditionally run together
        decision = supervisor.process_telemetry_window(rec, mode="SINGLE_LLM")
        decisions.append(decision.to_dict())

    return decisions
