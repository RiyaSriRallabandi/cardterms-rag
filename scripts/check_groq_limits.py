"""Report Groq quota state with a single cheap request.

The free tier limits requests and tokens, per minute and per day. A generation
run that trips the daily token limit fails identically to one that trips the
per-minute limit, nine minutes later. One tiny request reports which headroom
actually exists before an hour of work is committed to it.

    uv run python scripts/check_groq_limits.py
    uv run python scripts/check_groq_limits.py --model llama-3.1-8b-instant
"""

import argparse

import httpx

from cardterms.config import Settings
from cardterms.eval.llm import GROQ_ENDPOINT

INTERESTING = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
    "retry-after",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    args = parser.parse_args()

    settings = Settings()
    if not settings.groq_api_key:
        raise SystemExit("GROQ_API_KEY is not set in .env")

    response = httpx.post(
        GROQ_ENDPOINT,
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        json={
            "model": args.model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
        timeout=30,
    )

    print(f"\n  model   {args.model}")
    print(f"  status  {response.status_code}\n")
    for key in INTERESTING:
        if key in response.headers:
            print(f"    {key:34s} {response.headers[key]}")

    if response.status_code == 429:
        try:
            print(f"\n  error   {response.json()['error']['message']}")
        except Exception:  # noqa: BLE001 - fall back to the raw body
            print(f"\n  error   {response.text[:300]}")


if __name__ == "__main__":
    main()
