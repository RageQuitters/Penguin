"""
Chat Route — POST /api/chat
----------------------------
Powers the SentinelOps AI Agent chat interface.
The assistant has full awareness of the plant's live sensor context and
can answer questions about machines, faults, maintenance scheduling, etc.
Uses Pangu LLM (Huawei Cloud) with a rich system prompt.
"""
import json
from fastapi import APIRouter
from pydantic import BaseModel
from app.core.pangu_client import pangu_chat
from app.tools.data_store import get_all_machine_states

chat_router = APIRouter()


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    plant_id: str = "Jurong Plant A"


class ChatResponse(BaseModel):
    reply: str
    agent: str = "SentinelOps"


def _build_plant_context() -> str:
    """Build a live snapshot of plant state to inject into the system prompt."""
    try:
        machines = get_all_machine_states()
        lines = []
        critical = [m for m in machines if m.get("status") == "Critical"]
        warning = [m for m in machines if m.get("status") == "Warning"]
        normal = [m for m in machines if m.get("status") == "Normal"]

        lines.append(f"LIVE PLANT STATE — Jurong Plant A ({len(machines)} machines online):")
        lines.append(f"  Critical: {len(critical)}  Warning: {len(warning)}  Normal: {len(normal)}")

        if critical:
            lines.append("CRITICAL MACHINES:")
            for m in critical:
                faults = ", ".join(m.get("active_faults", [])) or "none"
                rul = m.get("rul_hours")
                lines.append(
                    f"  • {m['machine_id']} ({m['machine_type']}) — "
                    f"anomaly={m.get('anomaly_score', 0):.2f}, "
                    f"faults=[{faults}], "
                    f"RUL={f'{rul:.1f}h' if rul else 'N/A'}, "
                    f"tool_wear={m.get('tool_wear', 0):.0f}min"
                )

        if warning:
            lines.append("WARNING MACHINES:")
            for m in warning:
                faults = ", ".join(m.get("active_faults", [])) or "none"
                lines.append(
                    f"  • {m['machine_id']} ({m['machine_type']}) — "
                    f"anomaly={m.get('anomaly_score', 0):.2f}, faults=[{faults}]"
                )

        lines.append("ALL MACHINES SUMMARY:")
        for m in machines:
            rul_str = f"{m['rul_hours']:.1f}h" if m.get('rul_hours') else 'N/A'
            lines.append(
                f"  {m['machine_id']}: status={m.get('status','?')}, "
                f"score={m.get('anomaly_score',0):.2f}, "
                f"tool_wear={m.get('tool_wear',0):.0f}min, "
                f"RUL={rul_str}"
            )
        return "\n".join(lines)
    except Exception:
        return "Plant state unavailable."


SYSTEM_PROMPT = """You are SentinelOps, an industrial AI maintenance assistant powered by Huawei Cloud Pangu LLM.
You are monitoring Jurong Plant A, a precision manufacturing facility in Singapore.
You have four AI sub-agents: Anomaly Detector, Fault Classifier, Predictive Maintenance Agent, and yourself (Orchestrator / Pangu).

Your personality:
- Professional, concise, and direct — like an expert plant engineer
- Use precise numbers and technical language
- Highlight urgency clearly: CRITICAL → NORMAL
- Recommend specific actions (part numbers, time windows, team assignments)
- You can interpret sensor readings, explain fault codes (TWF, HDF, PWF, OSF, RNF), estimate maintenance windows

Fault code reference:
  TWF = Tool Wear Failure — tool insert past useful life
  HDF = Heat Dissipation Failure — cooling system fault, temperature delta too low at low RPM
  PWF = Power Failure — power draw outside envelope (torque × RPM)
  OSF = Overstrain Failure — high torque × tool wear combination
  RNF = Random Failure — stochastic failure unrelated to other sensors

Always reference the live plant context below when answering questions about specific machines.
If asked "which machine should I send engineers to" or similar triage questions, rank by: Critical status > highest anomaly score > lowest RUL.

{plant_context}
"""


@chat_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Multi-turn chat endpoint for the SentinelOps AI agent.
    Injects live plant context into the system prompt each turn.
    """
    plant_context = _build_plant_context()
    system = SYSTEM_PROMPT.format(plant_context=plant_context)

    # Build the conversation history as a single prompt (Pangu single-turn)
    # Format: inject all prior turns into the user prompt for context
    if len(request.messages) == 1:
        prompt = request.messages[0].content
    else:
        # Build multi-turn context
        history_lines = []
        for msg in request.messages[:-1]:
            role_label = "User" if msg.role == "user" else "SentinelOps"
            history_lines.append(f"{role_label}: {msg.content}")
        prompt = (
            "Conversation history:\n"
            + "\n".join(history_lines)
            + f"\n\nUser: {request.messages[-1].content}\n\nSentinelOps:"
        )

    reply = await pangu_chat(
        prompt=prompt,
        system=system,
        max_tokens=600,
        temperature=0.3,
    )

    return ChatResponse(reply=reply.strip())
