"""
Phase 3 Memory: OperationalMemory
Tracks short-term rolling machine states, alert transitions with hysteresis/cooldown,
and recent agent decision history to prevent alert flooding.
"""
from typing import Dict, List, Any, Optional
from collections import deque
import pandas as pd


class OperationalMemory:
    """Manages short-term operational state, alert state machine, and rolling memory."""

    def __init__(
        self,
        max_history: int = 100,
        cooldown_minutes: float = 60.0,
        hysteresis_ratio: float = 0.50,
    ):
        self.max_history = max_history
        self.cooldown_minutes = cooldown_minutes
        self.hysteresis_ratio = hysteresis_ratio

        self.history = deque(maxlen=max_history)
        self.decision_history = deque(maxlen=max_history)

        # State machine tracking
        self.current_alert_state: str = "NORMAL"  # NORMAL, WATCH, HIGH_RISK, CRITICAL
        self.last_escalation_time: Optional[pd.Timestamp] = None
        self.consecutive_high_risk_windows: int = 0
        self.consecutive_normal_windows: int = 0

    def update_state(
        self,
        timestamp: str,
        machine_state: str,
        anomaly_score: float,
        failure_prob: float,
        dominant_sensor: str,
    ) -> Dict[str, Any]:
        """Updates internal rolling history and executes state transition with hysteresis/cooldown."""
        ts = pd.to_datetime(timestamp)
        entry = {
            "timestamp": ts,
            "machine_state": machine_state,
            "anomaly_score": anomaly_score,
            "failure_prob": failure_prob,
            "dominant_sensor": dominant_sensor,
        }
        self.history.append(entry)

        # Cooldown check
        in_cooldown = False
        if self.last_escalation_time is not None:
            elapsed_min = (ts - self.last_escalation_time).total_seconds() / 60.0
            if 0 <= elapsed_min < self.cooldown_minutes:
                in_cooldown = True

        # State Machine Transitions
        prev_state = self.current_alert_state

        if machine_state in ("CRITICAL", "HIGH_RISK"):
            self.consecutive_high_risk_windows += 1
            self.consecutive_normal_windows = 0
        else:
            self.consecutive_normal_windows += 1
            self.consecutive_high_risk_windows = 0

        # Escalation / De-escalation
        if machine_state == "CRITICAL":
            self.current_alert_state = "CRITICAL"
            if prev_state != "CRITICAL" and not in_cooldown:
                self.last_escalation_time = ts
        elif machine_state == "HIGH_RISK":
            if self.current_alert_state != "CRITICAL":
                self.current_alert_state = "HIGH_RISK"
                if prev_state not in ("HIGH_RISK", "CRITICAL") and not in_cooldown:
                    self.last_escalation_time = ts
        elif machine_state == "WATCH":
            if self.current_alert_state in ("HIGH_RISK", "CRITICAL"):
                # Require de-escalation persistence
                if self.consecutive_normal_windows >= 5:
                    self.current_alert_state = "WATCH"
            else:
                self.current_alert_state = "WATCH"
        else:  # NORMAL
            if self.current_alert_state in ("HIGH_RISK", "CRITICAL", "WATCH"):
                if self.consecutive_normal_windows >= 10:
                    self.current_alert_state = "NORMAL"
            else:
                self.current_alert_state = "NORMAL"

        return {
            "previous_alert_state": prev_state,
            "current_alert_state": self.current_alert_state,
            "in_cooldown": in_cooldown,
            "consecutive_high_risk": self.consecutive_high_risk_windows,
            "consecutive_normal": self.consecutive_normal_windows,
        }

    def record_decision(self, decision_dict: Dict[str, Any]) -> None:
        """Stores a structured decision in recent memory."""
        self.decision_history.append(decision_dict)

    def get_recent_decisions(self, count: int = 5) -> List[Dict[str, Any]]:
        """Returns the N most recent decisions."""
        return list(self.decision_history)[-count:]

    def get_recent_states(self, count: int = 10) -> List[Dict[str, Any]]:
        """Returns the N most recent machine states."""
        return list(self.history)[-count:]
