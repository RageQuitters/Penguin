"""
Mock Data Store
---------------
Simulates queries to GaussDB (operational DB) and Cloud Stream Storage
(24h rolling buffer). In production, replace each function body with
a real DB query or API call.

Changes vs original:
  - All machines start as "Normal" status with low anomaly scores
  - A mutable _MACHINE_OVERRIDES dict lets the /api/breakdown endpoint
    force a specific machine into a broken-down (Critical) state
"""
import random
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

_MACHINE_SEEDS: dict[str, int] = {mid: i for i, mid in enumerate(MACHINE_PROFILES)}

# ---------------------------------------------------------------------------
# Mutable breakdown overrides — set via POST /api/breakdown
# ---------------------------------------------------------------------------

# machine_id -> True means the machine has been "broken down" by the operator
_MACHINE_OVERRIDES: dict[str, bool] = {}


def simulate_breakdown(machine_id: str) -> None:
    """Mark a machine as broken down (Critical). Called by the API route."""
    _MACHINE_OVERRIDES[machine_id] = True


def reset_breakdown(machine_id: str) -> None:
    """Clear a breakdown override, returning the machine to Normal."""
    _MACHINE_OVERRIDES.pop(machine_id, None)


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
        {"date": "2026-03-15", "work": "Routine inspection — no issues found",         "tech": "J. Lim"},
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
    "tool_insert_TI-330-A": 0,
    "bearing_SKF-6205":     7,
    "hydraulic_seal_HS-22": 2,
    "coolant_pump_CP-100":  1,
    "power_relay_PR-55":    4,
}

# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def query_baseline(machine_id: str, window_hours: int = 24) -> list[float]:
    seed = _MACHINE_SEEDS.get(machine_id, 0)
    rng = random.Random(seed)
    base = rng.uniform(0.05, 0.25)
    scores = []
    for i in range(window_hours):
        drift = (i / window_hours) * rng.uniform(0.0, 0.5)
        noise = rng.gauss(0, 0.03)
        scores.append(round(max(0.0, min(1.0, base + drift + noise)), 3))
    return scores


def get_machine_profile(machine_id: str) -> dict:
    profile = MACHINE_PROFILES.get(machine_id)
    if not profile:
        return {"type": "Unknown", "nominal_speed": 1500, "max_torque": 60, "max_wear": 200}
    return {"machine_id": machine_id, **profile}


def get_fault_history(machine_id: str, fault_type: str | None = None) -> list[dict]:
    history = FAULT_HISTORY.get(machine_id, [])
    if fault_type:
        history = [h for h in history if h["fault"] == fault_type]
    return history


def get_maintenance_log(machine_id: str) -> list[dict]:
    return MAINTENANCE_LOGS.get(machine_id, [])


def get_wear_trend(machine_id: str, current_wear: float, window_hours: int = 48) -> tuple:
    seed = _MACHINE_SEEDS.get(machine_id, 0)
    rng = random.Random(seed + 100)
    wear_per_hour = rng.uniform(2.0, 6.0)
    points = []
    for i in range(window_hours):
        offset = i - window_hours
        wear = max(0.0, current_wear + offset * wear_per_hour + rng.gauss(0, 0.5))
        points.append({"hour_offset": offset, "wear": round(wear, 1)})
    return points, round(wear_per_hour, 2)


def estimate_rul(current_wear: float, degradation_rate: float, threshold: float = 200.0) -> float:
    if degradation_rate <= 0:
        return 9999.0
    remaining = threshold - current_wear
    if remaining <= 0:
        return 0.0
    return round(remaining / degradation_rate, 2)


def check_parts_inventory(part_type: str) -> dict:
    for key, qty in PARTS_INVENTORY.items():
        if part_type.lower() in key.lower() or key.lower() in part_type.lower():
            return {"part": key, "quantity": qty, "available": qty > 0}
    return {"part": part_type, "quantity": 0, "available": False, "note": "Part not found in system"}


def get_all_machine_states() -> list[dict]:
    """
    Returns a snapshot of all machine states.

    Anomaly scores and fault types are determined by the real trained joblib
    models (Isolation Forest + Random Forest), using each machine's seeded
    sensor reading.  Machines in _MACHINE_OVERRIDES are shown as Critical
    (breakdown simulated) and bypass the ML scoring.
    """
    # Import here to avoid circular imports; lazy-load is safe because lru_cache
    # keeps the models in memory after the first call.
    from app.tools.ml_inference import run_isolation_forest, run_fault_classifier

    states = []
    for machine_id, profile in MACHINE_PROFILES.items():
        seed = _MACHINE_SEEDS[machine_id]
        rng = random.Random(seed + 999)

        if _MACHINE_OVERRIDES.get(machine_id):
            # Simulated breakdown — force Critical without bothering the models
            score  = round(rng.uniform(0.75, 0.97), 3)
            wear   = round(rng.uniform(185, 199), 1)
            status = "Critical"
            faults = ["TWF"]
        else:
            # Use the per-machine seeded reading from the Dashboard seeds
            wear = _MACHINE_DEFAULT_WEAR.get(machine_id, 50)
            reading = _MACHINE_DEFAULT_READINGS.get(machine_id, (298.1, 308.5, 1500, 42.0, wear))

            # Real ML inference
            score  = round(run_isolation_forest(reading), 4)
            faults = run_fault_classifier(reading)

            if score >= 0.7 or faults:
                status = "Critical"
            elif score >= 0.4:
                status = "Warning"
            else:
                status = "Normal"

        rul = round((200 - wear) / rng.uniform(2, 6), 1)
        states.append({
            "machine_id":    machine_id,
            "machine_type":  profile["type"],
            "status":        status,
            "anomaly_score": score,
            "rul_hours":     rul,
            "active_faults": faults,
            "tool_wear":     wear,
        })
    return states


# ---------------------------------------------------------------------------
# Default sensor readings for each machine (mirrors Dashboard seedReading)
# Used by get_all_machine_states() to feed real ML models.
# ---------------------------------------------------------------------------

_MACHINE_DEFAULT_READINGS: dict[str, tuple] = {
    "U-01": (298.2, 308.4, 1451, 38.2, 34),
    "U-02": (298.0, 308.2, 1502, 41.5, 56),
    "U-03": (299.1, 309.8, 1621, 68.4, 198),
    "U-04": (297.8, 307.9, 1398, 22.1, 12),
    "U-05": (298.5, 308.9, 1555, 47.3, 103),
    "U-06": (298.1, 308.3, 1483, 39.8, 29),
    "U-07": (298.1, 308.6, 1543, 42.8, 187),
    "U-08": (297.9, 308.0, 1449, 31.5, 8),
    "U-09": (298.3, 308.5, 1598, 45.1, 77),
    "U-10": (298.0, 308.1, 1552, 36.7, 41),
    "U-11": (298.6, 309.1, 1501, 53.2, 155),
    "U-12": (297.7, 307.8, 1402, 19.4, 22),
}

_MACHINE_DEFAULT_WEAR: dict[str, float] = {
    mid: reading[4] for mid, reading in _MACHINE_DEFAULT_READINGS.items()
}
