"""Minimal client for Groq's OpenAI-compatible chat completions endpoint.

Used for drafting evaluation questions and, later, for answer generation and
judging. Requests go over httpx rather than a vendor SDK to keep the
dependency surface small and the failure modes visible.
"""

import json
import time

import httpx

from cardterms.config import Settings

ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
TIMEOUT_SECONDS = 120

_settings = Settings()

MAX_RETRIES = 6
BACKOFF_BASE_SECONDS = 8
MAX_BACKOFF_SECONDS = 90


def _post_with_retry(payload: dict) -> httpx.Response:
    """POST with backoff on rate limiting.

    Free-tier limits are token-based, so bursts exhaust the budget well before
    the request-per-minute ceiling. The provider returns the wait it wants in
    the retry-after header; that is honoured when present.
    """
    headers = {"Authorization": f"Bearer {_settings.groq_api_key}"}
    response = None

    for attempt in range(MAX_RETRIES):
        response = httpx.post(
            ENDPOINT, headers=headers, json=payload, timeout=TIMEOUT_SECONDS
        )
        if response.status_code != 429:
            response.raise_for_status()
            return response

        wait = float(
            response.headers.get("retry-after", BACKOFF_BASE_SECONDS * (attempt + 1))
        )
        time.sleep(min(wait, MAX_BACKOFF_SECONDS))

    response.raise_for_status()
    return response


def complete(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    json_mode: bool = False,
) -> str:
    if not _settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    response = _post_with_retry(payload)
    return response.json()["choices"][0]["message"]["content"]


def complete_json(prompt: str, system: str = "", **kwargs) -> dict:
    raw = complete(prompt, system=system, json_mode=True, **kwargs)
    return json.loads(raw)
