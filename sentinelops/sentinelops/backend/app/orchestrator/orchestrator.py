"""
Orchestrator Agent (Pangu LLM)
-------------------------------
Coordinates all sub-agents with conditional routing. Uses TinyLlama
to justify decisions and synthesise the final work order.

Flow:
  1. Always call Anomaly Agent
  2. score < 0.4  → monitor only, stop
  3. score >= 0.4 → call Fault Agent
  4. faults OR score >= 0.7 → call Predictive Agent
  5. TinyLlama synthesises all results into a natural-language work order

Trace entries for LLM calls show the real TinyLlama response, not fake data.
The final synthesised work order is also returned in the decision so the
frontend can display it in the AI chat panel.
"""
import json
from app.core.pangu_client import pangu_chat
from app.core.trace import TraceLog
from app.models.schemas import (
    SensorReading, OrchestratorDecision, MachineStatus,
    Urgency, TraceAction, AnomalyResult, FaultResult, PredictiveResult,
)
from app.agents.anomaly_agent import run_anomaly_agent
from app.agents.fault_agent import run_fault_agent
from app.agents.predictive_agent import run_predictive_agent

AGENT_NAME = "Orchestrator"


async def run_orchestrator(machine_id: str, reading: SensorReading, trace: TraceLog) -> OrchestratorDecision:
    agents_called: list[str] = []
    anomaly_result: AnomalyResult | None = None
    fault_result: FaultResult | None = None
    predictive_result: PredictiveResult | None = None

    # --- Step 1: Anomaly Agent (always) ---
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.HANDOFF,
        input_data={"machine_id": machine_id, "reading": reading.model_dump()},
        explanation=f"Initiating analysis for {machine_id}. Dispatching to Anomaly Detection Agent.")
    anomaly_result = await run_anomaly_agent(machine_id, reading, trace)
    agents_called.append("Anomaly Detection Agent")
    score = anomaly_result.anomaly_score

    # --- Step 2: Route on anomaly score ---
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.REASONING,
        input_data={"anomaly_score": score, "threshold": 0.4},
        explanation=f"Evaluating routing. Score={score:.4f}. Fault agent threshold=0.4")

    if score < 0.4:
        # Ask TinyLlama to justify the skip — show real response in trace
        skip_reason = await _llm_justify_skip(machine_id, score, reading, trace)
        trace.add_entry(agent=AGENT_NAME, action=TraceAction.DECISION,
            output_data={"route": "MONITOR_ONLY", "reason": skip_reason},
            explanation=f"Score {score:.4f} < 0.4. Skipping Fault and Predictive agents.")
        return await _build_final(machine_id, reading, trace, agents_called,
                                   anomaly_result, None, None, skip_reason)

    # --- Step 3: Fault Agent ---
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.HANDOFF,
        input_data={"anomaly_score": score},
        explanation=f"Score {score:.4f} >= 0.4. Dispatching to Fault Classification Agent.")
    fault_result = await run_fault_agent(machine_id, reading, score, trace)
    agents_called.append("Fault Classification Agent")

    # --- Step 4: Route on faults or high score ---
    has_faults = bool(fault_result.active_faults)
    trace.add_entry(agent=AGENT_NAME, action=TraceAction.REASONING,
        input_data={"active_faults": fault_result.active_faults, "score": score, "threshold": 0.7},
        explanation=f"Evaluating Predictive Agent dispatch. Faults={fault_result.active_faults}, Score={score:.4f}")

    if has_faults or score >= 0.7:
        reason = "fault detected" if has_faults else f"high score ({score:.4f} >= 0.7)"
        trace.add_entry(agent=AGENT_NAME, action=TraceAction.HANDOFF,
            input_data={"reason": reason},
            explanation=f"Dispatching to Predictive Maintenance Agent ({reason}).")
        predictive_result = await run_predictive_agent(machine_id, reading, fault_result.active_faults, trace)
        agents_called.append("Predictive Maintenance Agent")
    else:
        trace.add_entry(agent=AGENT_NAME, action=TraceAction.DECISION,
            output_data={"route": "SKIP_PREDICTIVE"},
            explanation=f"No confirmed faults and score {score:.4f} < 0.7. Predictive agent not required.")

    return await _build_final(machine_id, reading, trace, agents_called,
                               anomaly_result, fault_result, predictive_result, None)


async def _build_final(
    machine_id, reading, trace, agents_called,
    anomaly, fault, predictive, routing_reason
) -> OrchestratorDecision:

    trace.add_entry(agent=AGENT_NAME, action=TraceAction.REASONING,
        input_data={"agents_called": agents_called},
        explanation="Sending all agent results to TinyLlama for final synthesis and work order generation")

    llm_result = await _llm_synthesise(machine_id, reading, anomaly, fault, predictive, routing_reason, trace)

    urgency = Urgency(llm_result.get("overall_urgency", "low"))
    if urgency in (Urgency.CRITICAL, Urgency.HIGH):
        status = MachineStatus.CRITICAL
    elif urgency == Urgency.MEDIUM:
        status = MachineStatus.WARNING
    else:
        status = MachineStatus.NORMAL

    decision = OrchestratorDecision(
        machine_id=machine_id, final_status=status, overall_urgency=urgency,
        work_order=llm_result.get("work_order", "No action required. Continue monitoring."),
        agents_called=agents_called,
        anomaly=anomaly, fault=fault, predictive=predictive,
    )

    trace.add_entry(agent=AGENT_NAME, action=TraceAction.DECISION,
        output_data={
            "status": status.value,
            "urgency": urgency.value,
            "work_order": decision.work_order,
            "agents": agents_called,
            "tinyllama_response": llm_result,
        },
        explanation=f"FINAL — Status: {status.value} | Urgency: {urgency.value} | Work order issued.")

    return decision


async def _llm_synthesise(machine_id, reading, anomaly, fault, predictive, routing_reason, trace) -> dict:
    parts = [f"Machine: {machine_id}"]
    parts.append(
        f"SENSOR READING: Air={reading.air_temperature}K, "
        f"ProcTemp={reading.process_temperature}K, "
        f"RPM={reading.rotational_speed}, "
        f"Torque={reading.torque}Nm, "
        f"Wear={reading.tool_wear}min"
    )
    if anomaly:
        parts.append(f"ANOMALY: score={anomaly.anomaly_score:.4f}, class={anomaly.classification.value}, urgency={anomaly.urgency.value}\n  {anomaly.reasoning}")
    if fault:
        parts.append(f"FAULTS: {fault.active_faults}, severity={fault.severity.value}, procurement={fault.procurement_flag}\n  {fault.reasoning}")
    if predictive:
        parts.append(f"PREDICTIVE: RUL={predictive.rul_hours}h, rate={predictive.degradation_rate}min/h, urgency={predictive.urgency.value}, procurement={predictive.procurement_flag}\n  {predictive.reasoning}")
    if routing_reason:
        parts.append(f"ROUTING: {routing_reason}")

    context = "\n".join(parts)

    prompt = f"""You are the Orchestrator of SentinelOps, an industrial predictive maintenance system powered by Huawei Cloud.

Intelligence gathered from sub-agents:
{context}

Produce a final maintenance decision:
1. Determine overall_urgency (low / medium / high / critical) — use the worst case from all agents
2. Write a professional, actionable work_order (2-5 sentences) for the plant operator
   - Mention: machine ID, fault types, estimated time to failure, parts needed, recommended action
   - If monitor only: say so clearly with next review interval
   - If immediate: be direct — specify what to do, which part, who should act

Respond ONLY in this JSON format:
{{"overall_urgency": "low"|"medium"|"high"|"critical", "work_order": "<operator work order text>"}}"""

    # Log the call to TinyLlama in the trace, including the machine context
    trace.add_entry(
        agent=AGENT_NAME,
        action=TraceAction.TOOL_CALL,
        input_data={
            "endpoint": "http://188.239.44.5:8000/generate",
            "model": "tinyllama",
            "machine_context": context,
        },
        explanation="Dispatching machine context to TinyLlama for final synthesis.",
    )

    raw = await pangu_chat(prompt, max_tokens=500)
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    result = json.loads(text.strip())

    # Record TinyLlama's real response in trace
    trace.add_entry(
        agent=AGENT_NAME,
        action=TraceAction.TOOL_CALL,
        output_data={"tinyllama_response": result},
        explanation=f"TinyLlama responded — urgency: {result.get('overall_urgency')}, work order generated.",
    )

    return result


async def _llm_justify_skip(machine_id: str, score: float, reading: SensorReading, trace: TraceLog) -> str:
    context = (
        f"Machine: {machine_id}\n"
        f"Sensor reading: Air={reading.air_temperature}K, "
        f"ProcTemp={reading.process_temperature}K, "
        f"RPM={reading.rotational_speed}, "
        f"Torque={reading.torque}Nm, "
        f"Wear={reading.tool_wear}min\n"
        f"Anomaly score: {score:.4f} (below 0.4 alert threshold)"
    )

    prompt = (
        f"{context}\n\n"
        f"In one sentence, state why no further agent analysis is needed "
        f"and what the recommended monitoring action is."
    )

    # Log the TinyLlama call with machine context in trace
    trace.add_entry(
        agent=AGENT_NAME,
        action=TraceAction.TOOL_CALL,
        input_data={
            "endpoint": "http://188.239.44.5:8000/generate",
            "model": "tinyllama",
            "machine_context": context,
        },
        explanation="Dispatching machine context to TinyLlama to justify monitor-only routing.",
    )

    raw = await pangu_chat(prompt, max_tokens=100)
    response = raw.strip()

    # Record real TinyLlama response in trace
    trace.add_entry(
        agent=AGENT_NAME,
        action=TraceAction.TOOL_CALL,
        output_data={"tinyllama_response": response},
        explanation=f"TinyLlama response: {response[:120]}{'…' if len(response) > 120 else ''}",
    )

    return response

