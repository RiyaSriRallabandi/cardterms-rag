"""Run retrieval over the evaluation set and record the result.

Each run stores its resolved configuration and the git commit that produced
it, so any reported number can be reproduced. Runs are never overwritten.

    uv run python scripts/run_eval.py --chunk-set fixed_512_ov0
    uv run python scripts/run_eval.py --mode dense --model bge-small \
        --chunk-set fixed_512_ov0 --note "dense baseline"
"""

import argparse
import hashlib
import json
import subprocess
import time

from tqdm import tqdm

from cardterms.config import ExperimentConfig
from cardterms.db import get_conn
from cardterms.eval.metrics import score_question
from cardterms.eval.relevance import relevant_chunks, relevant_documents
from cardterms.eval.stats import bootstrap_ci
from cardterms.logging import bind_run, configure_logging, log, new_run_id
from cardterms.retrieve.bm25 import BM25Retriever
from cardterms.retrieve.dense import DenseRetriever

# Questions with no labelled answer are scored in generation, not retrieval.
UNSCORED_CATEGORIES = ("unanswerable", "ambiguous")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001 - absence of git must not block a run
        return "unknown"


def chunk_set_name(config: ExperimentConfig) -> str:
    return (
        f"{config.chunking.strategy}_{config.chunking.chunk_tokens}"
        f"_ov{int(config.chunking.overlap_pct * 100)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--chunk-set", help="override the chunk set to evaluate")
    parser.add_argument("--top-k", type=int, help="override retrieval depth")
    parser.add_argument("--mode", default="bm25", choices=["bm25", "dense"])
    parser.add_argument("--model", help="embedding model key, required for dense")
    parser.add_argument("--no-prefix", action="store_true")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    if args.mode == "dense" and not args.model:
        raise SystemExit("--model is required in dense mode")

    configure_logging(json_output=False)
    config = ExperimentConfig.from_yaml(args.config)

    # Overrides are folded into the stored configuration, so the record
    # describes what actually ran rather than what the file said.
    chunk_set = args.chunk_set or chunk_set_name(config)
    if args.top_k:
        config.retrieval.top_k = args.top_k

    resolved = config.model_dump()
    resolved["chunk_set"] = chunk_set
    resolved["retrieval"]["mode"] = args.mode

    if args.mode == "dense":
        resolved["embedding"]["model_name"] = args.model
        resolved["embedding"]["query_prefix"] = "none" if args.no_prefix else "standard"
        run_name = f"dense_{args.model}_{chunk_set}"
        if args.no_prefix:
            run_name += "_noprefix"
    else:
        run_name = f"bm25_{chunk_set}"

    run_id = new_run_id()
    bind_run(run_id, chunk_set=chunk_set, mode=args.mode)

    depth = max(config.evaluation.k_values + [config.retrieval.top_k])
    per_question: list[dict] = []

    with get_conn() as conn:
        questions = conn.execute(
            "SELECT id, question_uid, question, category FROM eval_questions "
            "ORDER BY question_uid"
        ).fetchall()
        by_question = relevant_chunks(conn, chunk_set)
        docs_by_question = relevant_documents(conn)

        conn.execute(
            """
            INSERT INTO runs (id, run_name, git_sha, config, config_hash, notes)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                run_id,
                run_name,
                git_sha(),
                json.dumps(resolved),
                hashlib.sha256(
                    json.dumps(resolved, sort_keys=True).encode()
                ).hexdigest()[:16],
                args.note,
            ),
        )
        conn.commit()

        if args.mode == "dense":
            retriever = DenseRetriever(
                conn, chunk_set, args.model, use_prefix=not args.no_prefix
            )
        else:
            retriever = BM25Retriever.from_chunk_set(conn, chunk_set)

        # The dense retriever queries through this connection, so retrieval
        # runs inside the same block.
        for question in tqdm(questions, desc=run_name):
            if question["category"] in UNSCORED_CATEGORIES:
                continue

            started = time.perf_counter()
            results = retriever.search(question["question"], depth)
            latency_ms = (time.perf_counter() - started) * 1000

            chunk_ids = [chunk_id for chunk_id, _, _ in results]
            doc_ids = [doc_id for _, doc_id, _ in results]

            scores = score_question(
                chunk_ids,
                doc_ids,
                by_question.get(question["id"], {}),
                docs_by_question.get(question["id"], set()),
                config.evaluation.k_values,
            )

            per_question.append(
                {
                    "question_id": question["id"],
                    "category": question["category"],
                    "scores": scores,
                    "retrieved": [
                        {"chunk_id": c, "doc_id": d, "score": s, "rank": i}
                        for i, (c, d, s) in enumerate(results, start=1)
                    ],
                    "latency_ms": latency_ms,
                }
            )

    metric_names = sorted(per_question[0]["scores"]) if per_question else []
    aggregate = {}
    for name in metric_names:
        mean, low, high = bootstrap_ci(
            [row["scores"][name] for row in per_question],
            resamples=config.evaluation.bootstrap_samples,
        )
        aggregate[name] = {"mean": mean, "ci_low": low, "ci_high": high}

    by_category: dict[str, dict[str, float]] = {}
    for category in sorted({row["category"] for row in per_question}):
        rows = [r for r in per_question if r["category"] == category]
        by_category[category] = {
            "n": len(rows),
            **{
                name: sum(r["scores"][name] for r in rows) / len(rows)
                for name in metric_names
            },
        }

    with get_conn() as conn:
        for row in per_question:
            conn.execute(
                """
                INSERT INTO run_results
                    (run_id, question_id, retrieved, metrics, latencies)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    row["question_id"],
                    json.dumps(row["retrieved"]),
                    json.dumps(row["scores"]),
                    json.dumps({"retrieval_ms": row["latency_ms"]}),
                ),
            )
        conn.execute(
            "UPDATE runs SET finished_at = now(), status = 'complete', "
            "metrics = %s WHERE id = %s",
            (json.dumps({"aggregate": aggregate, "by_category": by_category}), run_id),
        )
        conn.commit()

    log.info("run_complete", run_id=run_id, questions=len(per_question))

    print(f"\n{'=' * 66}\n{run_name}   run {run_id[:8]}\n{'=' * 66}")
    print(f"{len(per_question)} scored questions\n")
    for name in ("hit_rate@5", "recall@5", "mrr", "doc_hit_rate@5", "ndcg@10"):
        if name in aggregate:
            value = aggregate[name]
            print(
                f"  {name:16s} {value['mean']:.3f}  "
                f"[{value['ci_low']:.3f}, {value['ci_high']:.3f}]"
            )
    print("\n  by category (hit_rate@5):")
    for category, values in by_category.items():
        print(f"    {category:20s} {values['hit_rate@5']:.3f}   n={values['n']}")


if __name__ == "__main__":
    main()
