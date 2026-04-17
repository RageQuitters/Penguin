"""
Fault Classification Agent
--------------------------
Uses Pangu LLM (Huawei Cloud) to enrich fault detections with context.
Context changes the response: e.g. PWF detected but power supply just
replaced → likely sensor artefact, not genuine power failure.

Tools:
  run_fault_classifier    → multi-label faults from Random Forest (ModelArts)
  get_fault_history       → fault event history (GaussDB)
  get_maintenance_log     → last maintenance dates + notes (GaussDB)
"""
import json
from app.core.pangu_client import pangu_chat
from app.core.trace import TraceLog
from app.models.schemas import SensorReading, FaultResult, Urgency, TraceAction
from app.tools.ml_inference import run_fault_classifier
from app.tools.data_store import get_fault_history, get_maintenance_log

AGENT_NAME = "Fault Classification Agent"
FAULT_EXPLANATIONS = {
    "TWF": "Tool Wear Failure — tool degraded past wear threshold",
    "HDF": "Heat Dissipation Failure — thermal management issue",
    "PWF": "Power Failure — power delivery instability",
    "OSF": "Overstrain Failure — torque/load exceeds operating limits",
    "RNF": "Random/Unknown Failure — unclassified anomaly",
}


async def run_fault_agent(machine_id: str, reading: SensorReading, anomaly_score: float, trace: TraceLog) -> FaultResult:

    # --- Tool: run_fault_classifier ---
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
        input_data={"tool": "run_fault_classifier", "reading": reading.to_tuple()},
        explanation="Running multi-label Random Forest fault classifier (via ModelArts or local joblib)")
    raw_faults = run_fault_classifier(reading.to_tuple())
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
        output_data={"detected_faults": raw_faults},
        explanation=f"Classifier raw output: {raw_faults if raw_faults else 'no faults detected'}")

    # --- Tool: get_fault_history (per detected fault) ---
    fault_history_map = {}
    for fault in raw_faults:
        trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
            input_data={"tool": "get_fault_history", "machine_id": machine_id, "fault": fault},
            explanation=f"Querying GaussDB fault_events for {fault} on {machine_id}")
        history = get_fault_history(machine_id, fault)
        fault_history_map[fault] = history
        trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
            output_data={"fault": fault, "count": len(history), "records": history},
            explanation=f"{fault}: {len(history)} prior events on {machine_id}")

    # --- Tool: get_maintenance_log ---
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
        input_data={"tool": "get_maintenance_log", "machine_id": machine_id},
        explanation="Querying GaussDB maintenance_log to contextualise fault detections")
    maint_log = get_maintenance_log(machine_id)
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
        output_data={"entries": maint_log, "count": len(maint_log)},
        explanation=f"Retrieved {len(maint_log)} maintenance entries for {machine_id}")

    # --- Pangu LLM Reasoning ---
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.REASONING,
        input_data={"raw_faults": raw_faults, "history": fault_history_map, "maint": maint_log},
        explanation="Sending fault detections + contextual history to Pangu LLM for enriched analysis")

    maint_summary = "\n".join(
        f"  {m['date']}: {m['work']} ({m['tech']})" for m in maint_log
    ) or "  No recent maintenance records."

    hist_summary = ""
    for f, recs in fault_history_map.items():
        actioned = sum(1 for r in recs if r.get("actioned"))
        hist_summary += f"\n  {f}: {len(recs)} events, {actioned} actioned"
    hist_summary = hist_summary or "\n  No prior fault history."

    fault_defs = "\n".join(f"  {k}: {v}" for k, v in FAULT_EXPLANATIONS.items())

    prompt = f"""You are the Fault Classification Agent in SentinelOps, an industrial predictive maintenance system.

Machine: {machine_id} | Anomaly Score: {anomaly_score:.4f}

SENSOR READING:
Air={reading.air_temperature}K, ProcTemp={reading.process_temperature}K, RPM={reading.rotational_speed}, Torque={reading.torque}Nm, Wear={reading.tool_wear}min

RAW FAULT CLASSIFIER OUTPUT (multi-label Random Forest): {raw_faults or 'None'}

FAULT TYPES:
{fault_defs}

FAULT HISTORY FOR DETECTED FAULTS:{hist_summary}

RECENT MAINTENANCE LOG:
{maint_summary}

INSTRUCTIONS:
For each detected fault, use maintenance context to determine if genuine or sensor artefact.
Key rules:
- If PWF detected but power supply was recently replaced → flag as probable sensor fault
- If TWF detected and wear > 180 min with prior TWF history → confirm genuine, escalate
- If HDF detected but coolant was recently serviced → re-examine sensor data first
- Overall severity = most critical confirmed genuine fault

Respond ONLY in this JSON format:
{{"active_faults": ["confirmed genuine fault codes only"], "severity": "low"|"medium"|"high"|"critical", "enriched_analysis": {{"FAULT_CODE": "explanation"}}, "procurement_flag": true|false, "reasoning": "<2-4 sentences>"}}"""

    raw = await pangu_chat(prompt, max_tokens=600)
    result = _parse_json(raw)

    trace.add_entry(agent=AGENT_NAME, action=TraceAction.DECISION, output_data=result,
        explanation=f"Confirmed faults: {result.get('active_faults', [])} | Severity: {result.get('severity')}")

    return FaultResult(
        active_faults=result.get("active_faults", []),
        severity=Urgency(result.get("severity", "low")),
        enriched_analysis=result.get("enriched_analysis", {}),
        reasoning=result.get("reasoning", ""),
        procurement_flag=result.get("procurement_flag", False),
    )


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())
