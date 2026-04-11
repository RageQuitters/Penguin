"""
Mock Data Store
---------------
Simulates queries to GaussDB (operational DB) and Cloud Stream Storage
(24h rolling buffer). In production, replace each function body with
a real DB query or API call.

Data is seeded with realistic industrial machine profiles and history.
"""
import random
import math
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Machine profiles (simulates GaussDB machines table)
# ---------------------------------------------------------------------------

MACHINE_PROFILES: dict[str, dict] = {
    "U-01": {"type": "Hydraulic Press",    "nominal_speed": 1450, "max_torque": 80,  "max_wear": 200},
    "U-02": {"type": "CNC Mill Type-A",    "nominal_speed": 1500, "max_torque": 60,  "max_wear": 200},
    "U-03": {"type": "Lathe Unit",         "nominal_speed": 1600, "max_torque": 70,  "max_wear": 200},
    "U-04": {"type": "Conveyor Drive",     "nominal_speed": 1400, "max_torque": 40,  "max_wear": 200},
    "U-05": {"type": "Pump Station",       "nominal_speed": 1550, "max_torque": 55,  "max_wear": 200},
    "U-06": {"type": "CNC Mill Type-B",    "nominal_speed": 1480, "max_torque": 65,  "max_wear": 200},
    "U-07": {"type": "CNC Mill Type-A",    "nominal_speed": 1500, "max_torque": 60,  "max_wear": 200},
    "U-08": {"type": "Hydraulic Press",    "nominal_speed": 1450, "max_torque": 80,  "max_wear": 200},
    "U-09": {"type": "Lathe Unit",         "nominal_speed": 1600, "max_torque": 70,  "max_wear": 200},
    "U-10": {"type": "Pump Station",       "nominal_speed": 1550, "max_torque": 55,  "max_wear": 200},
    "U-11": {"type": "CNC Mill Type-A",    "nominal_speed": 1500, "max_torque": 60,  "max_wear": 200},
    "U-12": {"type": "Conveyor Drive",     "nominal_speed": 1400, "max_torque": 40,  "max_wear": 200},
}

# Seed for reproducible mock scores per machine
_MACHINE_SEEDS: dict[str, int] = {mid: i for i, mid in enumerate(MACHINE_PROFILES)}

# ---------------------------------------------------------------------------
# Fault history (simulates GaussDB fault_events table)
# ---------------------------------------------------------------------------

FAULT_HISTORY: dict[str, list[dict]] = {
    "U-07": [
        {"fault": "TWF", "ts": "2026-04-08T10:30:00Z", "actioned": True,  "action_note": "Tool insert replaced"},
        {"fault": "TWF", "ts": "2026-04-10T22:15:00Z", "actioned": False, "action_note": None},
    ],
    "U-03": [
        {"fault": "HDF", "ts": "2026-04-09T14:00:00Z", "actioned": True,  "action_note": "Coolant flow restored"},
        {"fault": "PWF", "ts": "2026-04-10T08:30:00Z", "actioned": False, "action_note": None},
        {"fault": "TWF", "ts": "2026-04-11T06:00:00Z", "actioned": False, "action_note": None},
    ],
    "U-05": [
        {"fault": "OSF", "ts": "2026-04-07T16:45:00Z", "actioned": True,  "action_note": "Load rebalanced"},
    ],
}

# ---------------------------------------------------------------------------
# Maintenance logs (simulates GaussDB maintenance_log table)
# ---------------------------------------------------------------------------

MAINTENANCE_LOGS: dict[str, list[dict]] = {
    "U-07": [
        {"date": "2026-04-08", "work": "Replaced tool insert TI-440-B on spindle A", "tech": "B. Tan"},
        {"date": "2026-03-15", "work": "Routine inspection — no issues found", "tech": "J. Lim"},
    ],
    "U-03": [
        {"date": "2026-04-09", "work": "Restored coolant flow; cleared HDF alert", "tech": "A. Ng"},
    ],
    "U-01": [
        {"date": "2026-04-11", "work": "Scheduled maintenance completed — all sensors nominal", "tech": "B. Tan"},
    ],
}

# ---------------------------------------------------------------------------
# Parts inventory (simulates spare parts system)
# ---------------------------------------------------------------------------

PARTS_INVENTORY: dict[str, int] = {
    "tool_insert_TI-440-B": 3,
    "tool_insert_TI-330-A": 0,   # out of stock
    "bearing_SKF-6205":     7,
    "hydraulic_seal_HS-22": 2,
    "coolant_pump_CP-100":  1,
    "power_relay_PR-55":    4,
}

# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def query_baseline(machine_id: str, window_hours: int = 24) -> list[float]:
    """
    Fetch historical anomaly scores for the last `window_hours` hours.
    Simulates Cloud Stream Storage rolling buffer query.
    Returns a list of floats (hourly snapshots).
    """
    seed = _MACHINE_SEEDS.get(machine_id, 0)
    rng = random.Random(seed)
    # Generate a realistic trend: start low, may rise toward the end
    base = rng.uniform(0.05, 0.25)
    scores = []
    for i in range(window_hours):
        drift = (i / window_hours) * rng.uniform(0.0, 0.5)
        noise = rng.gauss(0, 0.03)
        scores.append(round(max(0.0, min(1.0, base + drift + noise)), 3))
    return scores


def get_machine_profile(machine_id: str) -> dict:
    """
    Retrieve machine type and expected operating ranges.
    Simulates GaussDB machines table lookup.
    """
    profile = MACHINE_PROFILES.get(machine_id)
    if not profile:
        return {"type": "Unknown", "nominal_speed": 1500, "max_torque": 60, "max_wear": 200}
    return {"machine_id": machine_id, **profile}


def get_fault_history(machine_id: str, fault_type: str | None = None) -> list[dict]:
    """
    Query fault event history for a machine, optionally filtered by type.
    Simulates GaussDB fault_events table.
    """
    history = FAULT_HISTORY.get(machine_id, [])
    if fault_type:
        history = [h for h in history if h["fault"] == fault_type]
    return history


def get_maintenance_log(machine_id: str) -> list[dict]:
    """
    Fetch the maintenance log for a machine.
    Simulates GaussDB maintenance_log table.
    """
    return MAINTENANCE_LOGS.get(machine_id, [])


def get_wear_trend(machine_id: str, current_wear: float, window_hours: int = 48) -> list[dict]:
    """
    Fetch tool wear readings over the last `window_hours`.
    Simulates a time-series query; fits a linear degradation model.
    Returns list of {hour_offset, wear} dicts plus degradation_rate (min/hour).
    """
    seed = _MACHINE_SEEDS.get(machine_id, 0)
    rng = random.Random(seed + 100)
    # Back-calculate starting wear
    wear_per_hour = rng.uniform(2.0, 6.0)
    points = []
    for i in range(window_hours):
        offset = i - window_hours
        wear = max(0.0, current_wear + offset * wear_per_hour + rng.gauss(0, 0.5))
        points.append({"hour_offset": offset, "wear": round(wear, 1)})
    return points, round(wear_per_hour, 2)


def estimate_rul(current_wear: float, degradation_rate: float, threshold: float = 200.0) -> float:
    """
    Compute Remaining Useful Life in hours.
    RUL = (threshold - current_wear) / degradation_rate
    """
    if degradation_rate <= 0:
        return 9999.0  # essentially infinite
    remaining = threshold - current_wear
    if remaining <= 0:
        return 0.0
    return round(remaining / degradation_rate, 2)


def check_parts_inventory(part_type: str) -> dict:
    """
    Query spare parts system for stock availability.
    Simulates integration with a parts management system.
    """
    # Try to find matching part (partial match)
    for key, qty in PARTS_INVENTORY.items():
        if part_type.lower() in key.lower() or key.lower() in part_type.lower():
            return {"part": key, "quantity": qty, "available": qty > 0}
    return {"part": part_type, "quantity": 0, "available": False, "note": "Part not found in system"}


def get_all_machine_states() -> list[dict]:
    """
    Returns a snapshot of all machine states (for dashboard polling).
    In production this reads from GaussDB current_states view.
    """
    states = []
    for machine_id, profile in MACHINE_PROFILES.items():
        seed = _MACHINE_SEEDS[machine_id]
        rng = random.Random(seed)
        score = rng.uniform(0.02, 0.95)
        wear = rng.uniform(5, 195)
        if score < 0.4:
            status = "Normal"
        elif score < 0.7:
            status = "Warning"
        else:
            status = "Critical"
        states.append({
            "machine_id": machine_id,
            "machine_type": profile["type"],
            "status": status,
            "anomaly_score": round(score, 3),
            "rul_hours": round((200 - wear) / rng.uniform(2, 6), 1),
            "active_faults": [],
            "tool_wear": round(wear, 1),
        })
    return states
