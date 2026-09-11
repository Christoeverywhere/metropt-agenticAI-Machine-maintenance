"""
Tool: get_risk_prediction
Reads frozen Phase 2D policy from checkpoints/phase2d/phase2d_frozen_policy.json
and evaluates impending failure risk.
"""
import os
import json
from typing import Dict, Any, Optional

try:
    from src.config import CHECKPOINT_DIR
except ImportError:
    from config import CHECKPOINT_DIR


_FROZEN_POLICY_CACHE: Optional[Dict[str, Any]] = None


def load_authoritative_frozen_policy(policy_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads authoritative frozen Phase 2D policy from disk (cached)."""
    global _FROZEN_POLICY_CACHE
    if _FROZEN_POLICY_CACHE is not None:
        return _FROZEN_POLICY_CACHE

    if policy_path is None or not os.path.exists(policy_path):
        policy_path = os.path.join(CHECKPOINT_DIR, "phase2d", "phase2d_frozen_policy.json")

    if os.path.exists(policy_path):
        with open(policy_path, "r") as f:
            _FROZEN_POLICY_CACHE = json.load(f)
    else:
        # Fallback default from Phase 2D validation
        _FROZEN_POLICY_CACHE = {
            "target_horizon_hours": 48.0,
            "frozen_policy": {
                "threshold": 0.020,
                "persistence": 20,
                "hysteresis_ratio": 0.50,
                "cooldown_minutes": 60.0,
            }
        }
    return _FROZEN_POLICY_CACHE


def get_risk_prediction(
    probability: float,
    current_persistence_windows: int = 0,
    is_in_active_alert: bool = False,
    policy_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluates Phase 2D probability against authoritative frozen operational policy.
    
    Returns:
        {
            "horizon_hours": 48,
            "probability": float,
            "entry_threshold": 0.020,
            "exit_threshold": 0.010,
            "persistence_required": 20,
            "persistence_met": bool,
            "policy_active": bool,
            "operational_alarm": bool,
            "risk_band": "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
        }
    """
    cfg = load_authoritative_frozen_policy(policy_path)
    policy = cfg.get("frozen_policy", {})

    entry_t = float(policy.get("threshold", 0.020))
    exit_ratio = float(policy.get("hysteresis_ratio", 0.50))
    exit_t = entry_t * exit_ratio
    p_req = int(policy.get("persistence", 20))
    cd_min = float(policy.get("cooldown_minutes", 60.0))

    p = float(probability)
    persist_met = bool(current_persistence_windows >= p_req)

    # Determine alarm activation with hysteresis
    if is_in_active_alert:
        alarm = bool(p >= exit_t and persist_met)
    else:
        alarm = bool(p >= entry_t and persist_met)

    # Risk Band
    if p < entry_t:
        band = "LOW"
    elif p < 0.10:
        band = "MODERATE"
    elif p < 0.50:
        band = "HIGH"
    else:
        band = "CRITICAL"

    return {
        "horizon_hours": 48,
        "probability": round(p, 4),
        "entry_threshold": round(entry_t, 4),
        "exit_threshold": round(exit_t, 4),
        "persistence_required_windows": p_req,
        "persistence_current_windows": current_persistence_windows,
        "persistence_met": persist_met,
        "cooldown_minutes": cd_min,
        "operational_alarm": alarm,
        "risk_band": band,
    }
