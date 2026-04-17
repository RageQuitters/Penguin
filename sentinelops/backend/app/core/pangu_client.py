"""
Local LLM Client (TinyLlama via custom /generate API)
----------------------------------------------------
Replaces Huawei Pangu with your self-hosted endpoint.

Endpoint:
  POST http://188.239.44.5:8000/generate

Headers:
  Authorization: Bearer mysecret123
  Content-Type: application/json
"""
import traceback
import json
import httpx
import asyncio
from app.core.config import get_settings

settings = get_settings()

# Change these if needed
LOCAL_LLM_API = "http://188.239.44.5:11434/api/generate"
LOCAL_LLM_KEY = "mysecret123"


async def pangu_chat(
    prompt: str,
    system: str = "You are SentinelOps, an industrial AI maintenance assistant. Keep responses to 3 sentences.",
    max_tokens: int = 100,
    temperature: float = 0.2,
) -> str:
    """
    Sends prompt to your qwen API.
    Falls back to mock response if API is unreachable.
    """

    # Convert chat → single prompt (your API is NOT chat-based)
    full_prompt = f"""{system}

Rules:
- Only provide actionable plant decisions

Task:
{prompt}

Answer:"""

    headers = {
        "Authorization": f"Bearer {LOCAL_LLM_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "qwen2.5:3b",
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                LOCAL_LLM_API,
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        # Adjust depending on your backend response format
        return (
            data.get("response")
        )

    except Exception as e:
        print("[LLM ERROR]")
        print("Status:", getattr(e.response, "status_code", None) if hasattr(e, "response") else None)
        print("Response:", getattr(e.response, "text", None) if hasattr(e, "response") else None)
        traceback.print_exc()
        return await _mock_chat_response(prompt)


# ---------------------------
# MOCK RESPONSES (UNCHANGED)
# ---------------------------

async def _mock_chat_response(prompt: str) -> str:
    """Mock SentinelOps chat replies for demo mode."""
    print("fake prompt mode")

    p = prompt.lower()

    if "which machine" in p or "send engineer" in p or "priority" in p or "triage" in p:
        return (
            "Based on live plant state, dispatch priority is:\n\n"
            "**1. U-07 (CRITICAL)** — Tool wear at 187 min (threshold: 200), anomaly score 0.82. "
            "TWF fault confirmed. RUL estimated at 4.2h. Send immediately.\n\n"
            "**2. U-03 (WARNING)** — Process temp elevated at 309.8 K, rotational speed 1621 rpm. "
            "Anomaly score 0.61. No confirmed faults yet but trajectory is deteriorating.\n\n"
            "**3. U-11 (WARNING)** — Tool wear 155 min, torque 53.2 Nm. Monitor closely.\n\n"
            "Recommend two-person team to U-07 now. Single technician to U-03 for inspection."
        )

    if "tool wear" in p or "twf" in p:
        return (
            "Tool Wear Failure (TWF) occurs when the tool insert exceeds its design lifetime. "
            "At Jurong Plant A, the critical threshold is 200 minutes of tool wear. "
            "U-07 is currently at 187 min — approximately 4–6 hours from failure. "
            "Action: Replace tool insert TI-440-B. Parts available."
        )

    if "fault" in p:
        return (
            "Active fault types:\n"
            "• TWF — U-07 approaching critical threshold\n"
            "• No other faults active\n"
        )

    if "rul" in p or "remaining useful life" in p:
        return (
            "RUL estimates:\n"
            "• U-07: ~4.2h (CRITICAL)\n"
            "• U-03: ~18.5h\n"
            "• U-11: ~31h\n"
        )

    if "maintenance" in p or "schedule" in p:
        return (
            "Maintenance schedule:\n"
            "• Immediate: U-07\n"
            "• Next shift: U-03\n"
            "• Monitor: U-11\n"
        )

    return "SentinelOps monitoring system active. U-07 is currently the highest priority."


# ---------------------------
# STRUCTURED MOCK (AGENTS)
# ---------------------------

async def _mock_structured_response(prompt: str) -> str:
    """
    Structured mock JSON responses for agent pipelines.
    """

    await asyncio.sleep(0.3)
    p = prompt.lower()

    # Anomaly agent
    if "classification" in p and "anomaly" in p:
        return json.dumps({
            "classification": "developing_fault",
            "urgency": "high",
            "reasoning": "Anomaly score indicates emerging fault pattern."
        })

    # Fault agent
    if "fault" in p:
        return json.dumps({
            "active_faults": ["TWF"],
            "severity": "high",
            "procurement_flag": True,
            "reasoning": "Tool wear fault detected."
        })

    # Predictive agent
    if "rul" in p:
        return json.dumps({
            "urgency": "high",
            "procurement_flag": True,
            "reasoning": "RUL below safe threshold."
        })

    # Orchestrator
    if "work_order" in p:
        return json.dumps({
            "overall_urgency": "medium",
            "work_order": "Inspect and replace tool soon."
        })

    return json.dumps({"result": "ok"})