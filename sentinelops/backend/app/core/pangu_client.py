"""
Huawei Cloud Pangu LLM Client
------------------------------
Wraps the Huawei ModelArts / Pangu LLM inference API.

Huawei Pangu exposes an OpenAI-compatible chat completions endpoint via
the ModelArts Dedicated Resource Pool or the Pangu API Gateway.

Endpoint pattern:
  POST {PANGU_API_BASE}/v1/chat/completions
  Headers:
    Authorization: Bearer {PANGU_API_KEY}
    X-Auth-Token: {IAM_TOKEN}          (alternative to Bearer)
    Content-Type: application/json

Auth:
  Option A — API Key (Bearer token, issued from Pangu API Gateway)
  Option B — IAM Token (short-lived, obtained via IAM /v3/auth/tokens)

This client supports both. Set PANGU_AUTH_MODE=apikey or iam in .env.

Reference:
  https://support.huaweicloud.com/intl/en-us/api-pangu/
  https://support.huaweicloud.com/intl/en-us/modelarts/
"""
import json
import httpx
import asyncio
from datetime import datetime, timezone, timedelta
from app.core.config import get_settings

settings = get_settings()

# Pangu model identifiers
# Use the model name configured in ModelArts deployment
PANGU_MODEL = "pangu-chat"   # override with PANGU_MODEL env var if needed


class HuaweiIAMAuth:
    """
    Obtains and caches a short-lived IAM token for Huawei Cloud API calls.
    Tokens are valid for 24 hours; this class refreshes automatically.
    """
    def __init__(self):
        self._token: str | None = None
        self._expires_at: datetime | None = None

    async def get_token(self) -> str:
        if self._token and self._expires_at and datetime.now(timezone.utc) < self._expires_at:
            return self._token
        await self._refresh()
        return self._token

    async def _refresh(self):
        payload = {
            "auth": {
                "identity": {
                    "methods": ["password"],
                    "password": {
                        "user": {
                            "name": settings.huawei_iam_username,
                            "password": settings.huawei_iam_password,
                            "domain": {"name": settings.huawei_iam_domain},
                        }
                    },
                },
                "scope": {"project": {"name": settings.huawei_region}},
            }
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"https://iam.{settings.huawei_region}.myhuaweicloud.com/v3/auth/tokens",
                json=payload,
            )
            resp.raise_for_status()
            self._token = resp.headers["X-Subject-Token"]
            # IAM tokens valid 24h; refresh 30min early
            self._expires_at = datetime.now(timezone.utc) + timedelta(hours=23, minutes=30)


# Module-level singleton for IAM token caching
_iam_auth = HuaweiIAMAuth()


async def pangu_chat(
    prompt: str,
    system: str = "You are SentinelOps, an industrial AI maintenance assistant.",
    max_tokens: int = 600,
    temperature: float = 0.2,
) -> str:
    """
    Send a chat message to the Pangu LLM and return the response text.

    Pangu uses an OpenAI-compatible /v1/chat/completions interface hosted on
    the Huawei ModelArts inference endpoint or the Pangu API Gateway.

    Falls back to a structured mock response if the endpoint is not configured,
    so development works without cloud credentials.
    """
    if not settings.pangu_api_base:
        return await _mock_pangu_response(prompt)

    headers = {"Content-Type": "application/json"}

    if settings.pangu_auth_mode == "apikey":
        headers["Authorization"] = f"Bearer {settings.pangu_api_key}"
    else:
        # IAM token auth
        token = await _iam_auth.get_token()
        headers["X-Auth-Token"] = token

    payload = {
        "model": settings.pangu_model or PANGU_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{settings.pangu_api_base}/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    # Standard OpenAI-compatible response shape
    return data["choices"][0]["message"]["content"].strip()


async def _mock_chat_response(prompt: str) -> str:
    """Mock SentinelOps chat replies for demo mode."""
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
            "U-07 is currently at 187 min — approximately 4–6 hours from failure at current usage rate. "
            "Action: Replace tool insert TI-440-B. Parts confirmed in-stock at Stores Bay 3."
        )
    if "fault" in p or "hdf" in p or "pwf" in p or "osf" in p:
        return (
            "Active fault types across Jurong Plant A:\n"
            "• **TWF** (Tool Wear Failure) — U-07: tool wear approaching critical threshold\n"
            "• No HDF, PWF, OSF, or RNF faults currently active\n\n"
            "To investigate a specific machine, run ANALYZE on that unit for a full agent pipeline report."
        )
    if "rul" in p or "remaining useful life" in p or "time to failure" in p:
        return (
            "Current RUL estimates (hours to predicted failure):\n"
            "• U-07: ~4.2h — CRITICAL, immediate action required\n"
            "• U-03: ~18.5h — elevated risk, schedule next maintenance window\n"
            "• U-11: ~31h — monitor, within acceptable range\n"
            "• All others: >72h — normal operating range\n\n"
            "RUL is computed by the Predictive Maintenance Agent using degradation rate models trained on historical plant data."
        )
    if "maintenance" in p or "schedule" in p or "upcoming" in p:
        return (
            "Recommended maintenance schedule (next 24 hours):\n\n"
            "**Immediate (0–4h):** U-07 — tool insert replacement (TWF fault active)\n"
            "**This shift (4–8h):** U-03 — thermal inspection and RPM calibration check\n"
            "**Next shift (8–16h):** U-11 — tool wear inspection, lubrication check\n"
            "**Routine (>16h):** U-05, U-09 — scheduled preventive maintenance per SOP\n\n"
            "All other units nominal — standard 72h monitoring interval applies."
        )
    if "handover" in p or "shift" in p or "report" in p:
        return (
            "**Shift Handover Report — Jurong Plant A**\n\n"
            "Machines online: 12 | Critical: 1 | Warning: 2 | Normal: 9\n\n"
            "Key events this shift:\n"
            "• U-07: TWF fault detected at 14:32. Tool wear 187/200 min. Escalated to maintenance team.\n"
            "• U-03: Anomaly score rose from 0.41 → 0.61. No fault confirmed yet. Keep under watch.\n"
            "• U-11: Torque trending upward (53.2 Nm vs baseline 45 Nm). Advisory issued.\n\n"
            "Incoming shift: Prioritise U-07 tool replacement. Review U-03 trend after 2 hours."
        )
    # Generic
    return (
        "SentinelOps is monitoring all 12 machines at Jurong Plant A. "
        "Current alert: U-07 has a Tool Wear Failure fault active with 4.2h estimated RUL. "
        "Ask me about specific machines, fault types, maintenance scheduling, or shift handover reports."
    )



    """
    Returns a structured mock JSON response when Pangu is not configured.
    Inspects the prompt to return the right schema for each agent type.
    Simulates realistic LLM latency.
    """
    await asyncio.sleep(0.3)  # simulate network round trip

    p = prompt.lower()

    # Anomaly agent response
    if '"classification"' in p or "classify this reading" in p or "classify" in p and "anomaly" in p:
        score_line = [l for l in prompt.split("\n") if "anomaly score:" in l.lower()]
        score = float(score_line[0].split(":")[-1].strip()) if score_line else 0.5
        if score < 0.4:
            cls, urg = "normal", "low"
            reason = f"Anomaly score {score:.4f} is well within normal operating bounds. The 24-hour trend shows no escalation. Machine operating nominally — no intervention required."
        elif score < 0.7:
            cls, urg = "transient_spike", "medium"
            reason = f"Anomaly score {score:.4f} is elevated but the 6-hour trend is relatively flat. This pattern is consistent with a transient operational spike rather than a developing fault. Recommend continued monitoring over the next shift."
        else:
            cls, urg = "developing_fault", "high"
            reason = f"Anomaly score {score:.4f} combined with a rising trend across the last 6 hours indicates a developing fault condition. Tool wear is approaching critical threshold. Immediate inspection recommended."
        return json.dumps({"classification": cls, "urgency": urg, "reasoning": reason})

    # Fault agent response
    if "fault classification" in p or "active_faults" in p or "enriched_analysis" in p:
        faults = []
        if "twf" in p and "tool wear" in p:
            faults = ["TWF"]
        return json.dumps({
            "active_faults": faults,
            "severity": "high" if faults else "low",
            "enriched_analysis": {
                f: f"{f} confirmed genuine based on sensor readings and fault history pattern. No recent maintenance addresses this specific fault type."
                for f in faults
            },
            "procurement_flag": bool(faults),
            "reasoning": f"Fault classifier detected {faults if faults else 'no active faults'}. Context from maintenance history and sensor values confirms assessment. {'Procurement of replacement parts recommended.' if faults else 'Machine appears fault-free at this time.'}",
        })

    # Predictive agent response
    if "remaining useful life" in p or "rul" in p or "procurement_flag" in p:
        rul_line = [l for l in prompt.split("\n") if "rul:" in l.lower() or "remaining useful life" in l.lower()]
        rul = 5.0
        for l in rul_line:
            parts = l.split(":")
            if len(parts) > 1:
                try:
                    rul = float(parts[-1].strip().split()[0])
                    break
                except ValueError:
                    pass
        urg = "critical" if rul < 2 else "high" if rul < 8 else "medium" if rul < 24 else "low"
        return json.dumps({
            "urgency": urg,
            "procurement_flag": rul < 8,
            "reasoning": f"RUL of {rul:.1f} hours at current degradation rate requires {'immediate' if rul < 2 else 'prompt' if rul < 8 else 'scheduled'} action. {'Parts availability is critical — escalate procurement immediately.' if rul < 8 else 'Schedule maintenance within the next maintenance window.'}",
        })

    # Orchestrator skip reasoning
    if "below 0.4" in p or "no further agent" in p:
        return "Anomaly score is below the 0.4 alert threshold, indicating normal operating conditions. Recommended action: log reading and continue routine monitoring — no maintenance intervention required."

    # Orchestrator synthesis
    if "work_order" in p or "synthesise" in p or "synthesize" in p:
        return json.dumps({
            "overall_urgency": "medium",
            "work_order": "SentinelOps Assessment: Elevated anomaly indicators detected. Review sensor trends over the next 4 hours. If tool wear continues to increase at the current rate, schedule tool insert replacement before the next shift. Parts TI-440-B are in stock. Assign to on-duty technician.",
        })

    # Chat / conversational fallback
    if "sentinelops:" in p or "conversation history" in p or "which machine" in p or "send engineers" in p:
        return _mock_chat_response(prompt)

    # Generic fallback
    return json.dumps({"result": "Analysis complete.", "reasoning": "No specific pattern matched in mock mode."})
