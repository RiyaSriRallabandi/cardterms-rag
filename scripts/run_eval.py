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
from cardterms.generate.answer import generate
from cardterms.generate.validate import validate
from cardterms.logging import bind_run, configure_logging, log, new_run_id
from cardterms.retrieve.bm25 import BM25Retriever
from cardterms.retrieve.dense import DenseRetriever
from cardterms.retrieve.diversify import diversify, select_by_entity
from cardterms.retrieve.entities import build_index, detect
from cardterms.retrieve.rerank import CrossEncoderReranker

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


def stratified(questions: list, limit: int) -> list:
    """Take `limit` questions, rotating across categories.

    A smoke test that samples in question order would draw entirely from one or
    two categories and miss the refusal questions, which are the ones a
    generation change is most likely to break.
    """
    buckets: dict[str, list] = {}
    for question in questions:
        buckets.setdefault(question["category"], []).append(question)

    picked = []
    while len(picked) < limit and any(buckets.values()):
        for category in sorted(buckets):
            if not buckets[category] or len(picked) >= limit:
                continue
            picked.append(buckets[category].pop(0))
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--chunk-set", help="override the chunk set to evaluate")
    parser.add_argument("--top-k", type=int, help="override retrieval depth")
    parser.add_argument("--mode", default="bm25", choices=["bm25", "dense"])
    parser.add_argument("--model", help="embedding model key, required for dense")
    parser.add_argument("--no-prefix", action="store_true")
    parser.add_argument("--note", default="")
    parser.add_argument("--rerank", choices=["ms-marco", "bge"])
    parser.add_argument(
        "--candidates",
        type=int,
        default=50,
        help="pool size retrieved before reranking",
    )
    parser.add_argument("--augment-rerank", action="store_true")
    parser.add_argument(
        "--max-per-doc",
        type=int,
        help="cap passages from any one document within the top-k window",
    )
    parser.add_argument(
        "--entity-select",
        action="store_true",
        help="reserve top-k slots for each card the question names",
    )
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "groq"])
    parser.add_argument("--gen-model", help="override the generator model")
    parser.add_argument("--prompt", default="answer_v1")
    parser.add_argument(
        "--clarify-gate",
        action="store_true",
        help="ask which card when the question names none, without generating",
    )
    parser.add_argument(
        "--only",
        help="comma-separated categories, e.g. ambiguous,unanswerable",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="evaluate a stratified subset; marks the run as a smoke test",
    )
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

    if args.rerank:
        resolved["reranking"]["enabled"] = True
        resolved["reranking"]["model_name"] = args.rerank
        resolved["reranking"]["candidate_pool"] = args.candidates
        resolved["reranking"]["augmented"] = args.augment_rerank
        run_name += f"_rr-{args.rerank}-{args.candidates}"
        if args.augment_rerank:
            run_name += "-aug"

    if args.max_per_doc:
        resolved["retrieval"]["max_per_doc"] = args.max_per_doc
        run_name += f"_cap{args.max_per_doc}"

    if args.entity_select:
        resolved["retrieval"]["entity_select"] = True
        run_name += "_ent"

    # A partial run must never be mistaken for a full one in the runs table.
    if args.only or args.limit:
        run_name += "_smoke"

    if args.generate:
        resolved["generation"]["provider"] = args.provider
        resolved["generation"]["model_name"] = args.gen_model or "default"
        resolved["generation"]["prompt_version"] = args.prompt
        resolved["generation"]["clarify_gate"] = args.clarify_gate
        run_name += f"_gen-{args.provider}"
        if args.clarify_gate:
            run_name += "-gate"

    run_id = new_run_id()
    bind_run(run_id, chunk_set=chunk_set, mode=args.mode)

    depth = max(config.evaluation.k_values + [config.retrieval.top_k])
    per_question: list[dict] = []

    with get_conn() as conn:
        questions = conn.execute(
            "SELECT id, question_uid, question, category, reference_answer "
            "FROM eval_questions ORDER BY question_uid"
        ).fetchall()
        if args.only:
            wanted = {name.strip() for name in args.only.split(",")}
            questions = [q for q in questions if q["category"] in wanted]
        if args.limit:
            questions = stratified(questions, args.limit)
        if not questions:
            raise SystemExit("no questions selected")

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

        reranker = (
            CrossEncoderReranker(conn, args.rerank, augment=args.augment_rerank)
            if args.rerank
            else None
        )

        entity_index = (
            build_index(conn) if (args.entity_select or args.clarify_gate) else {}
        )

        # The dense retriever queries through this connection, so retrieval
        # runs inside the same block.
        for question in tqdm(questions, desc=run_name):
            scored = question["category"] not in UNSCORED_CATEGORIES
            if not scored and not args.generate:
                continue

            started = time.perf_counter()
            if reranker:
                pool = retriever.search(question["question"], args.candidates)
                results = reranker.rerank(question["question"], pool, depth)
            else:
                results = retriever.search(question["question"], depth)
            if args.entity_select:
                results = select_by_entity(
                    results,
                    config.retrieval.top_k,
                    detect(question["question"], entity_index),
                )
            if args.max_per_doc:
                results = diversify(results, config.retrieval.top_k, args.max_per_doc)
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

            answer = None
            checks: dict = {}
            if args.generate:
                answer = generate(
                    conn,
                    question["question"],
                    results[: config.retrieval.top_k],
                    prompt_version=args.prompt,
                    provider=args.provider,
                    model=args.gen_model,
                    max_context_tokens=config.generation.max_context_tokens,
                    entity_index=entity_index if args.clarify_gate else None,
                )
                checks = validate(answer, question["reference_answer"])

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
                    "scored": scored,
                    "answer": answer.text if answer else None,
                    "abstained": answer.abstained if answer else None,
                    "abstention_kind": answer.abstention_kind if answer else None,
                    "cited": answer.cited if answer else [],
                    "checks": checks,
                }
            )

    # Retrieval metrics average only over questions that have labels.
    # Unanswerable and ambiguous questions are generated for but not scored on
    # retrieval, and including them would depress every aggregate by a fifth.
    retrieval_rows = [r for r in per_question if r["scored"]]

    metric_names = sorted(retrieval_rows[0]["scores"]) if retrieval_rows else []
    aggregate = {}
    for name in metric_names:
        mean, low, high = bootstrap_ci(
            [row["scores"][name] for row in retrieval_rows],
            resamples=config.evaluation.bootstrap_samples,
        )
        aggregate[name] = {"mean": mean, "ci_low": low, "ci_high": high}

    by_category: dict[str, dict[str, float]] = {}
    for category in sorted({row["category"] for row in retrieval_rows}):
        rows = [r for r in retrieval_rows if r["category"] == category]
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
                    (run_id, question_id, retrieved, answer, citations,
                     abstained, metrics, latencies)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    row["question_id"],
                    json.dumps(row["retrieved"]),
                    row.get("answer"),
                    json.dumps(row.get("cited", [])),
                    row.get("abstained"),
                    json.dumps({**row["scores"], **row.get("checks", {})}),
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
    print(f"{len(retrieval_rows)} scored questions of {len(per_question)} run\n")
    for name in (
        "hit_rate@5",
        "evidence@5",
        "recall@5",
        "mrr",
        "doc_hit_rate@5",
        "ndcg@10",
    ):
        if name in aggregate:
            value = aggregate[name]
            print(
                f"  {name:16s} {value['mean']:.3f}  "
                f"[{value['ci_low']:.3f}, {value['ci_high']:.3f}]"
            )
    print("\n  by category (hit_rate@5 / evidence@5):")
    for category, values in by_category.items():
        print(
            f"    {category:20s} {values['hit_rate@5']:.3f}   "
            f"{values.get('evidence@5', float('nan')):.3f}   n={values['n']}"
        )

    if args.generate:
        answerable = [r for r in per_question if r["scored"]]
        refusable = [r for r in per_question if not r["scored"]]

        false_abstain = sum(1 for r in answerable if r["abstained"])
        correct_abstain = sum(1 for r in refusable if r["abstained"])
        bad_citations = sum(
            1 for r in answerable if not r["checks"].get("citations_valid", True)
        )
        uncited = sum(r["checks"].get("uncited_claims", 0) for r in answerable)
        figure_checked = [
            r
            for r in answerable
            if r["checks"].get("contains_reference_figure") is not None
        ]
        figure_hits = sum(
            1 for r in figure_checked if r["checks"]["contains_reference_figure"]
        )

        print("\n  generation")
        print(f"    correct abstention   {correct_abstain}/{len(refusable)}")
        print(f"    false abstention     {false_abstain}/{len(answerable)}")
        print(f"    invalid citations    {bad_citations} answer(s)")
        print(f"    uncited claims       {uncited} sentence(s)")
        if figure_checked:
            print(f"    expected figure present {figure_hits}/{len(figure_checked)}")


if __name__ == "__main__":
    main()
