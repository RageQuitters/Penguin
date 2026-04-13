"""
OpenRouter Client
-----------------
Wraps OpenRouter's OpenAI-compatible chat completions API.
OpenRouter proxies many models (OpenAI, Anthropic, Meta, etc.) through a
single OpenAI-compatible interface, so the request shape is identical.

Endpoint:
  POST https://openrouter.ai/api/v1/chat/completions
  Headers:
    Authorization: Bearer {OPENROUTER_API_KEY}
    Content-Type: application/json
    HTTP-Referer: <your site>   (optional, for OpenRouter leaderboard)
    X-Title: SentinelOps        (optional)
"""
import httpx
import asyncio
from app.core.config import get_settings

settings = get_settings()

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "openai/gpt-oss-120b:free"


async def openai_chat(
    prompt: str,
    system: str = "You are SentinelOps, an industrial AI maintenance assistant.",
    max_tokens: int = 600,
    temperature: float = 0.2,
) -> str:
    """
    Send a chat message via OpenRouter and return the response text.
    Falls back to mock mode if OPENAI_API_KEY is not configured.
    """
    if not settings.openai_api_key:
        return await _mock_openai_response(prompt)

    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://sentinelops.local",  # optional
        "X-Title": "SentinelOps",                      # optional
    }

    payload = {
        "model": settings.openai_model or OPENROUTER_DEFAULT_MODEL,
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
            f"{OPENROUTER_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    return data["choices"][0]["message"]["content"].strip()


async def _mock_openai_response(prompt: str) -> str:
    await asyncio.sleep(0.2)
    return f"[MOCK OpenRouter] Received {len(prompt)} chars. Configure OPENAI_API_KEY to enable live calls."