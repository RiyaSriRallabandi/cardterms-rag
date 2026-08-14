"""Text generation behind a single interface.

Two providers are supported so that a local model and a hosted one can be
compared without changing anything else: the local path runs entirely on the
machine, the hosted path is faster and stronger but rate-limited. Requests go
over httpx rather than vendor SDKs to keep failure modes visible.
"""

import json
import time

import httpx

from cardterms.config import Settings

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"

DEFAULT_MODELS = {"groq": "llama-3.3-70b-versatile", "ollama": "llama3.2:3b"}

TIMEOUT_SECONDS = 180
MAX_RETRIES = 6
BACKOFF_BASE_SECONDS = 8
MAX_BACKOFF_SECONDS = 90

_settings = Settings()


def _groq(messages: list[dict], model: str, temperature: float, json_mode: bool) -> str:
    if not _settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env")

    payload: dict = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {"Authorization": f"Bearer {_settings.groq_api_key}"}
    response = None

    for attempt in range(MAX_RETRIES):
        response = httpx.post(
            GROQ_ENDPOINT, headers=headers, json=payload, timeout=TIMEOUT_SECONDS
        )
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        wait = float(
            response.headers.get("retry-after", BACKOFF_BASE_SECONDS * (attempt + 1))
        )
        time.sleep(min(wait, MAX_BACKOFF_SECONDS))

    response.raise_for_status()
    return ""


def _ollama(messages: list[dict], model: str, temperature: float) -> str:
    response = httpx.post(
        OLLAMA_ENDPOINT,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def complete(
    prompt: str,
    system: str = "",
    provider: str = "groq",
    model: str | None = None,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    model = model or DEFAULT_MODELS[provider]
    if provider == "ollama":
        return _ollama(messages, model, temperature)
    return _groq(messages, model, temperature, json_mode)


def complete_json(prompt: str, system: str = "", **kwargs) -> dict:
    return json.loads(complete(prompt, system=system, json_mode=True, **kwargs))
