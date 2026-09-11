"""
Phase 3 UI: Interactive Diagnostic Dashboard
Provides CLI rendering and Web dashboard server for real-time inspection of Agentic Decisions across scenarios.
"""
import os
import sys
import json
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.phase3.evaluation.scenarios import build_scenario_dataset
from src.phase3.agents.supervisor import SupervisorAgent


def render_cli_dashboard(decision: Dict[str, Any], scenario_name: str = ""):
    """Renders terminal diagnostic card matching Section 22 dashboard layout."""
    m_state = decision.get("machine_state", "NORMAL")
    pred = decision.get("prediction", {})
    anom = decision.get("anomaly", {})
    diag = decision.get("diagnosis", {})
    hist = decision.get("historical_context", {})
    maint = decision.get("maintenance", {})
    evidence = decision.get("evidence", [])

    print("+" + "-" * 58 + "+")
    print(f"| {'AGENTIC PREDICTIVE MAINTENANCE DECISION SYSTEM':^56} |")
    if scenario_name:
        print(f"| {scenario_name:^56} |")
    print("+" + "-" * 58 + "+")
    print(f"| Machine State:       {m_state:<37} |")
    print(f"|                                                          |")
    print(f"| 48h Failure Risk:    {pred.get('failure_probability', 0.0)*100:>6.2f}% ({pred.get('risk_level', 'LOW'):<8})                 |")
    print(f"| Anomaly Score:       {anom.get('score', 0.0):>6.2f}                             |")
    print(f"| Persistence:         {anom.get('persistent', False)!s:<15}                     |")
    print(f"|                                                          |")
    print(f"| Dominant Indicator:  {diag.get('primary_indicator', 'None'):<37} |")
    print(f"| Pattern Type:        {diag.get('pattern_type', 'NOMINAL'):<37} |")
    print("+" + "-" * 58 + "+")
    print(f"| HISTORICAL MATCH                                         |")
    sim_name = hist.get('similar_failure', 'None')
    sim_score = hist.get('similarity', 0.0) * 100.0
    print(f"| * Reference:         {sim_name:<37} |")
    print(f"| * Similarity:        {sim_score:>5.1f}%                              |")
    print("+" + "-" * 58 + "+")
    print(f"| AGENT MAINTENANCE RECOMMENDATION                         |")
    print(f"| * Action:            {maint.get('action', 'MONITOR'):<37} |")
    print(f"| * Urgency:           {maint.get('urgency', 'LOW'):<37} |")
    print(f"| * Subsystem:         {maint.get('recommended_area', 'General')[:37]:<37} |")
    print("+" + "-" * 58 + "+")
    print(f"| EVIDENCE TRACE                                           |")
    for ev in evidence[:4]:
        clean_ev = ev[:52]
        print(f"| [x] {clean_ev:<52} |")
    print("+" + "-" * 58 + "+\n")


def main():
    scenarios = build_scenario_dataset()
    supervisor = SupervisorAgent()

    print("\n" + "=" * 60)
    print("DEMO: AGENTIC PREDICTIVE MAINTENANCE DASHBOARD")
    print("=" * 60 + "\n")

    for sc in scenarios:
        rec = sc["record"]
        decision = supervisor.process_telemetry_window(rec, mode="AGENTIC")
        render_cli_dashboard(decision.to_dict(), scenario_name=sc["name"])


if __name__ == "__main__":
    main()
