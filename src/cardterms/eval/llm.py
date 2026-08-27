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

DEFAULT_MODELS = {"groq": "openai/gpt-oss-20b", "ollama": "llama3.2:3b"}

# gpt-oss models emit a reasoning trace before their answer. The grounding
# contract asks for a short quote-backed sentence, and a visible chain of
# thought both breaks that format and inflates latency, so reasoning is held
# at the lowest setting the provider allows.
REASONING_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")

TIMEOUT_SECONDS = 180
MAX_RETRIES = 6
BACKOFF_BASE_SECONDS = 8
MAX_BACKOFF_SECONDS = 90

# Free-tier tokens per minute, per model. The generation prompt carries five
# passages and runs to roughly 3,000 tokens, so a run issuing requests as fast
# as they complete exceeds the window within seconds and then waits out reset
# periods longer than any sane backoff. Pacing client-side is cheaper than
# retrying, and keeps a full evaluation inside the free tier.
MODEL_TPM = {
    "openai/gpt-oss-20b": 8_000,
    "openai/gpt-oss-120b": 8_000,
}
DEFAULT_TPM = 6_000
TPM_HEADROOM = 0.85
CHARS_PER_TOKEN = 4

_settings = Settings()

# (timestamp, estimated tokens) for requests inside the current window.
_recent: list[tuple[float, int]] = []

# Seconds the last call spent waiting on the rate limit rather than working.
# Reported separately from latency: a request that waited 50 seconds for a
# token budget is not a slow request, it is a throttled one, and conflating
# the two makes the system look an order of magnitude slower than it is.
_last_throttle = 0.0


def last_throttle_seconds() -> float:
    return _last_throttle


def _pace(model: str, estimated: int) -> None:
    """Block until `estimated` tokens fit inside this model's per-minute budget."""
    global _last_throttle
    budget = int(MODEL_TPM.get(model, DEFAULT_TPM) * TPM_HEADROOM)
    started = time.time()

    while True:
        cutoff = time.time() - 60
        _recent[:] = [entry for entry in _recent if entry[0] > cutoff]
        used = sum(tokens for _, tokens in _recent)
        if used + estimated <= budget or not _recent:
            break
        # Wait for the oldest request to age out of the window.
        time.sleep(max(0.5, 60 - (time.time() - _recent[0][0])))

    _last_throttle = time.time() - started
    _recent.append((time.time(), estimated))


def _groq(messages: list[dict], model: str, temperature: float, json_mode: bool) -> str:
    if not _settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env")

    payload: dict = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if model in REASONING_MODELS:
        payload["reasoning_effort"] = "low"

    headers = {"Authorization": f"Bearer {_settings.groq_api_key}"}
    response = None

    estimated = sum(len(m["content"]) for m in messages) // CHARS_PER_TOKEN
    _pace(model, estimated)

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
        # A per-minute limit is worth waiting out; a per-day limit is not, and
        # retrying it silently burns the remaining attempts for no reason.
        if wait > MAX_BACKOFF_SECONDS:
            raise RuntimeError(
                f"Groq rate limit needs {wait:.0f}s to reset — "
                f"{_limit_detail(response)}"
            )
        time.sleep(wait)

    raise RuntimeError(
        f"Groq rate limited after {MAX_RETRIES} attempts — {_limit_detail(response)}"
    )


def _limit_detail(response: httpx.Response) -> str:
    """Which quota was exhausted, as reported by the API."""
    try:
        message = response.json()["error"]["message"]
    except Exception:  # noqa: BLE001 - diagnostics must not mask the real error
        message = response.text[:200]
    remaining = {
        key: value
        for key, value in response.headers.items()
        if key.startswith("x-ratelimit")
    }
    return f"{message} | {remaining}"


def _ollama(messages: list[dict], model: str, temperature: float) -> str:
    global _last_throttle
    _last_throttle = 0.0  # nothing to throttle locally

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
