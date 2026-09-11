"""
Phase 3 Agent: MonitoringAgent
Responsible for generating machine state, severity classification, and anomaly/risk trends
directly from deterministic tool outputs.
"""
from typing import Dict, Any, Optional
from src.phase3.schemas.schemas import MachineStateOutput
from src.phase3.tools.machine_state_tool import get_current_machine_state


class MonitoringAgent:
    """Agent that creates the operational machine state from Phase 1 and Phase 2D outputs."""

    def __init__(self, frozen_policy: Optional[Dict[str, Any]] = None):
        self.frozen_policy = frozen_policy

    def assess_state(self, record: Dict[str, Any]) -> MachineStateOutput:
        """Evaluates machine condition deterministically without hallucination."""
        return get_current_machine_state(record, frozen_policy=self.frozen_policy)
