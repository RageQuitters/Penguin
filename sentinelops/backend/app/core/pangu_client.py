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


async def _mock_pangu_response(prompt: str) -> str:
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

    # Generic fallback
    return json.dumps({"result": "Analysis complete.", "reasoning": "No specific pattern matched in mock mode."})
