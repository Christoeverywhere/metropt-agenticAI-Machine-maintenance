"""
Tool: generate_maintenance_context
Maps sensor attribution and subsystem anomalies to operational maintenance inspection areas,
procedural guidelines, and subsystem boundaries.
"""
from typing import Dict, List, Any, Optional

SUBSYSTEM_MAPPINGS = {
    "DV_pressure": {
        "subsystem": "Pneumatic Discharge & Pressure Regulation",
        "components": ["Discharge check valve", "Pressure relief valve", "Dryer purge line", "Discharge pipe union"],
        "recommended_inspection": "Inspect pneumatic discharge lines, check valve seating, and pressure relief valve integrity for air leakage.",
        "standard_safety_note": "Depressurize compressor reservoir before physical inspection.",
    },
    "H1": {
        "subsystem": "Air Intake & Pneumatic Pre-compression",
        "components": ["Air intake filter", "Intake valve", "H1 pressure transducer line"],
        "recommended_inspection": "Inspect air intake filter for clogging or particulate restriction; verify H1 line sealing.",
        "standard_safety_note": "Lock-out tag-out compressor drive unit prior to filter housing disassembly.",
    },
    "TP2": {
        "subsystem": "Compressor Intermediate / Intercooler Stage",
        "components": ["Intercooler pipework", "Intermediate temperature/pressure manifold"],
        "recommended_inspection": "Check intermediate pipework and heat exchanger connections for thermal/pressure leaks.",
        "standard_safety_note": "Allow thermal cooling before touching intercooler manifold surfaces.",
    },
    "TP3": {
        "subsystem": "High-Pressure Delivery & Post-Cooler",
        "components": ["Aftercooler line", "Delivery manifold", "Check valves"],
        "recommended_inspection": "Inspect high-pressure delivery manifold and check valve seals.",
        "standard_safety_note": "Verify zero pressure gauge reading before uncoupling manifold fittings.",
    },
    "Reservoirs": {
        "subsystem": "Main Air Storage & Condensate Drain",
        "components": ["Main air reservoir", "Auto-drain valve", "Drain solenoid"],
        "recommended_inspection": "Inspect main reservoir seals, pressure sensor fittings, and condensate drain valve operation.",
        "standard_safety_note": "Drain all condensed moisture and isolate pressure tank.",
    },
    "Motor_current": {
        "subsystem": "Electromechanical Drive & Motor",
        "components": ["Induction motor stator", "VFD inverter / power supply", "Coupling shaft"],
        "recommended_inspection": "Check motor winding resistance, shaft coupling alignment, and power supply balance.",
        "standard_safety_note": "Electrically isolate 400V 3-phase supply and discharge capacitor banks.",
    },
    "Oil_temperature": {
        "subsystem": "Lubrication & Thermal Management",
        "components": ["Oil radiator", "Thermostatic valve", "Lubrication oil level/filter"],
        "recommended_inspection": "Check lubrication oil level, viscosity, oil filter differential pressure, and radiator fan operation.",
        "standard_safety_note": "Hot oil hazard - do not open oil sump while unit is at operating temperature.",
    },
}


def generate_maintenance_context(
    dominant_sensor: str,
    supporting_sensors: Optional[List[str]] = None,
    machine_state: str = "NORMAL",
) -> Dict[str, Any]:
    """Generates deterministic subsystem context, inspection targets, and procedural guidelines
    grounded strictly in the active sensor indicators.
    """
    supporting = supporting_sensors or []
    dom_info = SUBSYSTEM_MAPPINGS.get(
        dominant_sensor,
        {
            "subsystem": "General Air Compressor Subsystem",
            "components": ["Compressor package"],
            "recommended_inspection": "Perform general visual inspection of compressor package.",
            "standard_safety_note": "Follow standard facility lock-out tag-out safety protocols.",
        },
    )

    all_components = list(dom_info["components"])
    all_subsystems = [dom_info["subsystem"]]

    for s in supporting:
        if s in SUBSYSTEM_MAPPINGS and s != dominant_sensor:
            supp_info = SUBSYSTEM_MAPPINGS[s]
            all_subsystems.append(supp_info["subsystem"])
            all_components.extend(supp_info["components"])

    return {
        "primary_subsystem": dom_info["subsystem"],
        "related_subsystems": list(set(all_subsystems)),
        "candidate_inspection_components": list(set(all_components)),
        "recommended_inspection_protocol": dom_info["recommended_inspection"],
        "safety_protocol": dom_info["standard_safety_note"],
        "machine_state_context": machine_state,
    }
