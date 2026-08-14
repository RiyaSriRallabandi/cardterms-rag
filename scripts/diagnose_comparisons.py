"""Check whether multi-document questions received all the documents they need.

hit_rate@k treats a question as answered if any relevant chunk appears. A
comparison needs a figure from each card, so a run can score a hit while giving
the generator only half the evidence. This measures document *coverage* — the
fraction of a question's gold documents present in the passages the generator
actually saw.

    uv run python scripts/diagnose_comparisons.py
    uv run python scripts/diagnose_comparisons.py --top-k 10
"""

import argparse
import json

from cardterms.db import get_conn
from cardterms.eval.relevance import relevant_chunks, relevant_documents


def evidence_by_document(conn, chunk_set: str) -> dict[int, dict[int, set[int]]]:
    """Return {question_id: {doc_id: {chunk_id}}} for labelled chunks.

    Document coverage only asks whether the right agreement appeared. This asks
    the sharper question: did a passage carrying the labelled span appear, for
    each document the question needs?
    """
    graded = relevant_chunks(conn, chunk_set)
    all_ids = {cid for grades in graded.values() for cid in grades}
    if not all_ids:
        return {}

    owner = {
        row["id"]: row["doc_id"]
        for row in conn.execute(
            "SELECT id, doc_id FROM chunks WHERE id = ANY(%s)", (list(all_ids),)
        ).fetchall()
    }

    resolved: dict[int, dict[int, set[int]]] = {}
    for question_id, grades in graded.items():
        per_doc = resolved.setdefault(question_id, {})
        for chunk_id in grades:
            per_doc.setdefault(owner[chunk_id], set()).add(chunk_id)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", help="defaults to the most recent generation run")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="passages handed to the generator; coverage is measured over these",
    )
    args = parser.parse_args()

    with get_conn() as conn:
        if args.run_id:
            run = conn.execute(
                "SELECT id, run_name, config FROM runs WHERE id::text LIKE %s",
                (args.run_id + "%",),
            ).fetchone()
        else:
            run = conn.execute(
                "SELECT id, run_name, config FROM runs WHERE run_name LIKE '%%gen-%%' "
                "ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        if not run:
            raise SystemExit("no generation run found")

        config = run["config"]
        if isinstance(config, str):
            config = json.loads(config)
        chunk_set = config["chunk_set"]

        docs_by_question = relevant_documents(conn)
        evidence = evidence_by_document(conn, chunk_set)
        rows = conn.execute(
            """
            SELECT q.id, q.question_uid, q.category, r.retrieved,
                   r.answer, r.abstained
            FROM run_results r
            JOIN eval_questions q ON q.id = r.question_id
            WHERE r.run_id = %s
            ORDER BY q.category, q.question_uid
            """,
            (run["id"],),
        ).fetchall()

    print(
        f"\n{'=' * 78}\ndocument coverage @{args.top_k}: {run['run_name']}\n{'=' * 78}"
    )

    doc_cov: dict[str, list[float]] = {}
    ev_cov: dict[str, list[float]] = {}
    multi_doc = []

    for row in rows:
        gold = docs_by_question.get(row["id"], set())
        if not gold:
            continue

        retrieved = row["retrieved"]
        if isinstance(retrieved, str):
            retrieved = json.loads(retrieved)
        window = retrieved[: args.top_k]
        seen_docs = {p["doc_id"] for p in window}
        seen_chunks = {p["chunk_id"] for p in window}

        per_doc = evidence.get(row["id"], {})
        with_evidence = {
            doc_id for doc_id in gold if per_doc.get(doc_id, set()) & seen_chunks
        }

        doc_cov.setdefault(row["category"], []).append(
            len(gold & seen_docs) / len(gold)
        )
        ev_cov.setdefault(row["category"], []).append(len(with_evidence) / len(gold))

        if len(gold) > 1:
            # For each document still missing its evidence, the best rank at
            # which one of its labelled chunks appears anywhere in the ranking.
            missing_at = []
            for doc_id in sorted(gold - with_evidence):
                ranks = [
                    p["rank"]
                    for p in retrieved
                    if p["chunk_id"] in per_doc.get(doc_id, set())
                ]
                missing_at.append(min(ranks) if ranks else None)

            multi_doc.append(
                {
                    "uid": row["question_uid"],
                    "need": len(gold),
                    "docs": len(gold & seen_docs),
                    "evidence": len(with_evidence),
                    "abstained": row["abstained"],
                    "missing_at": missing_at,
                }
            )

    print("\n  coverage by category (document / evidence)")
    for category in sorted(doc_cov):
        d, e = doc_cov[category], ev_cov[category]
        print(
            f"    {category:20s} doc {sum(d) / len(d):.3f}  "
            f"evidence {sum(e) / len(e):.3f}   "
            f"complete {sum(1 for v in e if v == 1.0)}/{len(e)}"
        )

    print(f"\n  questions needing more than one document (n={len(multi_doc)})")
    print(f"    {'question':38s} need docs evid  abst  evidence missing until rank")
    for m in multi_doc:
        flag = "yes " if m["abstained"] else "no  "
        at = ", ".join("never" if r is None else str(r) for r in m["missing_at"]) or "-"
        print(
            f"    {m['uid']:38s} {m['need']:4d} {m['docs']:4d} {m['evidence']:4d}  "
            f"{flag}  {at}"
        )

    incomplete = [m for m in multi_doc if m["evidence"] < m["need"]]
    reachable = [m for m in incomplete if all(r is not None for r in m["missing_at"])]
    print(
        f"\n  {len(incomplete)} incomplete; {len(reachable)} of those have every "
        f"missing passage somewhere in the retrieved pool"
    )


if __name__ == "__main__":
    main()
