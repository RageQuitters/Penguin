"""
Predictive Maintenance Agent
-----------------------------
Uses Pangu LLM (Huawei Cloud) to determine true maintenance urgency,
combining RUL, degradation rate, and spare parts availability.

Key: if RUL=3h but part is out of stock → CRITICAL, not just HIGH.

Tools:
  get_wear_trend          → 48h wear time series + degradation rate (GaussDB)
  estimate_rul            → Remaining Useful Life in hours
  check_parts_inventory   → spare parts stock levels
"""
import json
from app.core.pangu_client import pangu_chat
from app.core.trace import TraceLog
from app.models.schemas import SensorReading, PredictiveResult, Urgency, TraceAction
from app.tools.data_store import get_wear_trend, estimate_rul, check_parts_inventory

AGENT_NAME = "Predictive Maintenance Agent"
FAULT_PART_MAP = {
    "TWF": "tool_insert_TI-440-B",
    "HDF": "coolant_pump_CP-100",
    "PWF": "power_relay_PR-55",
    "OSF": "bearing_SKF-6205",
    "RNF": "bearing_SKF-6205",
}


async def run_predictive_agent(machine_id: str, reading: SensorReading, active_faults: list[str], trace: TraceLog) -> PredictiveResult:

    # --- Tool: get_wear_trend ---
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
        input_data={"tool": "get_wear_trend", "machine_id": machine_id, "window": "48h"},
        explanation="Fetching 48h tool wear time series from GaussDB to compute degradation rate")
    trend_points, degradation_rate = get_wear_trend(machine_id, reading.tool_wear, window_hours=48)
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
        output_data={"degradation_rate": degradation_rate, "recent_points": trend_points[-4:]},
        explanation=f"Tool wear degrading at {degradation_rate} min/hour")

    # --- Tool: estimate_rul ---
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
        input_data={"tool": "estimate_rul", "current_wear": reading.tool_wear, "rate": degradation_rate, "threshold": 200.0},
        explanation="Computing RUL = (200 - current_wear) / degradation_rate")
    rul = estimate_rul(reading.tool_wear, degradation_rate, threshold=200.0)
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
        output_data={"rul_hours": rul}, explanation=f"Estimated RUL: {rul}h at current wear rate")

    # --- Tool: check_parts_inventory ---
    parts_to_check = list({FAULT_PART_MAP.get(f, "tool_insert_TI-440-B") for f in active_faults}) or ["tool_insert_TI-440-B"]
    parts_availability: dict[str, bool] = {}
    parts_details: list[dict] = []

    for part in parts_to_check:
        trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
            input_data={"tool": "check_parts_inventory", "part": part},
            explanation=f"Checking spare parts system for: {part}")
        inv = check_parts_inventory(part)
        parts_availability[inv.get("part", part)] = inv.get("available", False)
        parts_details.append(inv)
        trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
            output_data=inv,
            explanation=f"{inv.get('part')}: {'IN STOCK' if inv.get('available') else 'OUT OF STOCK'} (qty={inv.get('quantity', 0)})")

    # --- Pangu LLM Reasoning ---
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.REASONING,
        input_data={"rul": rul, "rate": degradation_rate, "parts": parts_availability, "faults": active_faults},
        explanation="Sending RUL, degradation rate, and parts data to Pangu LLM for urgency decision")

    parts_summary = "\n".join(
        f"  {p.get('part')}: qty={p.get('quantity', 0)}, available={p.get('available')}"
        for p in parts_details
    )

    prompt = f"""You are the Predictive Maintenance Agent in SentinelOps, an industrial predictive maintenance system.

Machine: {machine_id}

TOOL WEAR STATUS:
- Current wear: {reading.tool_wear} min (threshold: 200 min)
- Degradation rate: {degradation_rate} min/hour
- Estimated RUL: {rul} hours

ACTIVE CONFIRMED FAULTS: {active_faults or 'None'}

SPARE PARTS INVENTORY:
{parts_summary}

Determine true maintenance urgency using ALL factors:

Urgency rules:
- RUL < 2h → at minimum HIGH; if any part unavailable → CRITICAL
- RUL 2–8h → MEDIUM; if any part unavailable → HIGH
- RUL 8–24h → LOW to MEDIUM depending on fault severity
- RUL > 24h → LOW (unless multiple critical faults)
- Parts unavailability escalates urgency by one level
- Set procurement_flag=true when urgency HIGH or CRITICAL AND any part is unavailable

Respond ONLY in this JSON format:
{{"urgency": "low"|"medium"|"high"|"critical", "procurement_flag": true|false, "reasoning": "<2-4 sentences combining RUL, rate, and parts>"}}"""

    raw = await pangu_chat(prompt, max_tokens=400)
    result = _parse_json(raw)

    trace.add_entry(agent=AGENT_NAME, action=TraceAction.DECISION, output_data=result,
        explanation=f"Urgency: {result.get('urgency')} | Procurement flag: {result.get('procurement_flag')}")

    return PredictiveResult(
        rul_hours=rul,
        degradation_rate=degradation_rate,
        parts_available=parts_availability,
        urgency=Urgency(result.get("urgency", "medium")),
        procurement_flag=result.get("procurement_flag", False),
        reasoning=result.get("reasoning", ""),
    )


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())
