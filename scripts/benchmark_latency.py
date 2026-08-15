"""Measure per-stage latency on the served pipeline.

Accuracy is measured on the evaluation set; latency has to be measured on the
service, because the service is what a user waits for. Timings are taken from
the same code path that answers requests rather than from a reimplementation.

The first call loads model weights onto the GPU and is discarded: reporting a
cold start as typical latency overstates it several times over.

    uv run python scripts/benchmark_latency.py
    uv run python scripts/benchmark_latency.py --provider ollama --model llama3.2:3b
"""

import argparse
import statistics
import time

from cardterms.db import get_conn
from cardterms.logging import configure_logging
from cardterms.service import CardTerms

WARMUP = 2


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(q * (len(ordered) - 1)))
    return ordered[index]


def report(name: str, values: list[float]) -> None:
    print(
        f"    {name:10s} p50 {statistics.median(values):7.0f} ms"
        f"   p95 {percentile(values, 0.95):7.0f} ms"
        f"   min {min(values):6.0f}   max {max(values):6.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="questions to time")
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    args = parser.parse_args()

    configure_logging(json_output=False)

    # Real evaluation questions, so context sizes match production conditions.
    with get_conn() as conn:
        questions = [
            row["question"]
            for row in conn.execute(
                """
                SELECT question FROM eval_questions
                WHERE category NOT IN ('ambiguous')
                ORDER BY question_uid
                LIMIT %s
                """,
                (args.n + WARMUP,),
            ).fetchall()
        ]

    started = time.perf_counter()
    service = CardTerms(provider=args.provider, model=args.model)
    startup = time.perf_counter() - started

    retrieve: list[float] = []
    rerank: list[float] = []
    generate: list[float] = []
    total: list[float] = []
    throttle: list[float] = []

    for index, question in enumerate(questions):
        result = service.ask(question)
        if index < WARMUP or result.abstained:
            continue
        retrieve.append(result.timing.retrieve)
        rerank.append(result.timing.rerank)
        generate.append(result.timing.generate)
        total.append(result.timing.total)
        throttle.append(result.timing.throttle)
        print(f"  [{len(total):3d}] {result.timing.total:7.0f} ms  {question[:60]}")

    service.close()

    if not total:
        raise SystemExit("no timed questions; every question abstained")

    print(f"\n{'=' * 66}")
    print(f"latency: {args.provider} / {args.model}")
    print(f"{'=' * 66}")
    print(f"  cold start {startup:.1f} s   ({WARMUP} warmup calls discarded)")
    print(f"  {len(total)} questions timed\n")
    report("retrieve", retrieve)
    report("rerank", rerank)
    report("generate", generate)
    report("total", total)

    waited = [value for value in throttle if value > 1.0]
    if waited:
        print(
            f"\n  rate limiting: {len(waited)} of {len(total)} requests waited, "
            f"median {statistics.median(waited) / 1000:.1f} s"
        )
        print("  (excluded from latency; it is a throughput cap, not compute time)")

    share = {
        "retrieve": statistics.median(retrieve),
        "rerank": statistics.median(rerank),
        "generate": statistics.median(generate),
    }
    whole = sum(share.values()) or 1.0
    print("\n  share of median request")
    for stage, value in share.items():
        print(f"    {stage:10s} {value / whole:5.1%}")


if __name__ == "__main__":
    main()
