"""
Anomaly Detection Agent
-----------------------
Uses Pangu LLM (Huawei Cloud) to reason about sensor anomalies.

Tools:
  run_isolation_forest   → anomaly score (ML)
  query_baseline         → 24h historical trend (Cloud Stream Storage / GaussDB)
  get_machine_profile    → machine type + operating ranges (GaussDB)
"""
import json
from app.core.pangu_client import pangu_chat
from app.core.trace import TraceLog
from app.models.schemas import (
    SensorReading, AnomalyResult, AnomalyClassification, Urgency, TraceAction,
)
from app.tools.ml_inference import run_isolation_forest
from app.tools.data_store import query_baseline, get_machine_profile

AGENT_NAME = "Anomaly Detection Agent"


async def run_anomaly_agent(machine_id: str, reading: SensorReading, trace: TraceLog) -> AnomalyResult:

    # --- Tool: run_isolation_forest ---
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
        input_data={"tool": "run_isolation_forest", "reading": reading.to_tuple()},
        explanation="Scoring sensor reading against trained Isolation Forest (via ModelArts or local joblib)")
    anomaly_score = run_isolation_forest(reading.to_tuple())
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
        output_data={"anomaly_score": anomaly_score},
        explanation=f"Isolation Forest anomaly score: {anomaly_score:.4f}")

    # --- Tool: query_baseline ---
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
        input_data={"tool": "query_baseline", "machine_id": machine_id, "window": "24h"},
        explanation="Querying 24h historical anomaly scores from Cloud Stream Storage")
    baseline = query_baseline(machine_id, window_hours=24)
    recent = baseline[-6:]
    trend = "rising" if recent[-1] > recent[0] else "stable or falling"
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
        output_data={"last_6h": recent, "trend": trend},
        explanation=f"24h trend: {trend}. Last 6h scores: {recent}")

    # --- Tool: get_machine_profile ---
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
        input_data={"tool": "get_machine_profile", "machine_id": machine_id},
        explanation="Fetching machine type and operating ranges from GaussDB")
    profile = get_machine_profile(machine_id)
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.TOOL_CALL,
        output_data=profile, explanation=f"Machine type: {profile.get('type')}")

    # --- Pangu LLM Reasoning ---
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.REASONING,
        input_data={"score": anomaly_score, "trend": trend, "recent": recent, "profile": profile},
        explanation="Sending tool outputs to Pangu LLM for contextual anomaly classification")

    prompt = f"""You are the Anomaly Detection Agent in SentinelOps, an industrial predictive maintenance system.

Machine: {machine_id} ({profile.get('type', 'Unknown')})

SENSOR READING:
- Air Temperature: {reading.air_temperature} K
- Process Temperature: {reading.process_temperature} K
- Rotational Speed: {reading.rotational_speed} RPM
- Torque: {reading.torque} Nm
- Tool Wear: {reading.tool_wear} min

ISOLATION FOREST ANOMALY SCORE: {anomaly_score:.4f} (0=normal, 1=most anomalous)

24H BASELINE TREND: {baseline}
LAST 6H: {recent} — trend is {trend}

MACHINE PROFILE: Type={profile.get('type')}, NominalSpeed={profile.get('nominal_speed')} RPM, MaxTorque={profile.get('max_torque')} Nm, WearThreshold={profile.get('max_wear')} min

Classify this reading. Rules:
- score < 0.4 → "normal", urgency "low"
- high score but flat 6h trend → "transient_spike", urgency "medium"
- rising trend AND elevated score → "developing_fault", urgency "high" or "critical"
- tool wear > 180 min always escalates urgency

Respond ONLY in this JSON format:
{{"classification": "normal"|"transient_spike"|"developing_fault", "urgency": "low"|"medium"|"high"|"critical", "reasoning": "<2-4 sentences>"}}"""

    raw = await pangu_chat(prompt, max_tokens=400)
    result = _parse_json(raw)

    trace.add_entry(agent=AGENT_NAME, action=TraceAction.DECISION, output_data=result,
        explanation=f"Pangu classified: {result.get('classification')} / urgency: {result.get('urgency')}")

    return AnomalyResult(
        anomaly_score=anomaly_score,
        classification=AnomalyClassification(result["classification"]),
        urgency=Urgency(result["urgency"]),
        reasoning=result["reasoning"],
        baseline_trend=baseline,
    )


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())
