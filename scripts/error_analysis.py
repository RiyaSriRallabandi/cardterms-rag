"""Classify every question by what went wrong, and where.

An aggregate score says how much is broken. It does not say which component to
fix. Each question carries a retrieval outcome and an answer verdict, and the
combination identifies the cause: a wrong answer with complete evidence in the
context is a generation failure, the same wrong answer with no relevant passage
retrieved is a retrieval failure, and the two have nothing to do with each
other.

Requires a generation run and its judged verdicts:

    uv run python scripts/judge_run.py --run-id 86582a6b
    uv run python scripts/error_analysis.py --run-id 86582a6b
"""

import argparse
import json
from pathlib import Path

from cardterms.db import get_conn

JUDGEMENT_DIR = Path("data/eval/judgements")
UNSCORED_CATEGORIES = ("unanswerable", "ambiguous")

# Ordered so the report reads from success to worst failure.
BUCKETS = (
    ("correct", "answered correctly"),
    ("correct_refusal", "correctly refused (no answer in corpus)"),
    ("partial_evidence_gap", "partial answer, evidence incomplete"),
    ("partial_despite_evidence", "partial answer despite complete evidence"),
    ("refused_evidence_gap", "refused, evidence incomplete — defensible"),
    ("refused_despite_evidence", "refused despite complete evidence"),
    ("wrong_unretrievable", "wrong: no relevant passage exists at depth 50"),
    ("wrong_not_promoted", "wrong: passage in the pool, not in the top 5"),
    ("wrong_evidence_gap", "wrong: answered on incomplete evidence"),
    ("wrong_grounding", "wrong: evidence was complete — grounding failure"),
    ("confabulation", "answered a question the corpus cannot answer"),
)


def classify(row: dict, verdict: str | None) -> str:
    unlabelled = row["category"] in UNSCORED_CATEGORIES
    if unlabelled:
        return "correct_refusal" if row["abstained"] else "confabulation"

    complete = (row["evidence"] or 0.0) >= 1.0

    if row["abstained"]:
        return "refused_evidence_gap" if not complete else "refused_despite_evidence"

    if verdict == "correct":
        return "correct"
    if verdict == "partial":
        return "partial_evidence_gap" if not complete else "partial_despite_evidence"

    # Wrong, or ungraded: separate retrieval causes from generation causes.
    if not row["hit5"]:
        return "wrong_not_promoted" if row["hit50"] else "wrong_unretrievable"
    return "wrong_evidence_gap" if not complete else "wrong_grounding"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt", default="judge_v1")
    args = parser.parse_args()

    path = JUDGEMENT_DIR / f"{args.run_id[:8]}_{args.prompt}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} — run judge_run.py first")
    verdicts = {
        record["question_uid"]: record["verdict"]
        for record in (
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    }

    with get_conn() as conn:
        run = conn.execute(
            "SELECT id, run_name FROM runs WHERE id::text LIKE %s",
            (args.run_id + "%",),
        ).fetchone()
        if not run:
            raise SystemExit(f"no run matching {args.run_id!r}")

        rows = conn.execute(
            """
            SELECT q.question_uid, q.category, q.question,
                   r.answer, coalesce(r.abstained, false) AS abstained,
                   (r.metrics->>'evidence@5')::float      AS evidence,
                   (r.metrics->>'hit_rate@5')::float      AS hit5,
                   (r.metrics->>'hit_rate@50')::float     AS hit50
            FROM run_results r
            JOIN eval_questions q ON q.id = r.question_id
            WHERE r.run_id = %s
            ORDER BY q.category, q.question_uid
            """,
            (run["id"],),
        ).fetchall()

    classified: dict[str, list[dict]] = {name: [] for name, _ in BUCKETS}
    for row in rows:
        record = dict(row)
        bucket = classify(record, verdicts.get(row["question_uid"]))
        record["verdict"] = verdicts.get(row["question_uid"])
        classified[bucket].append(record)

    total = len(rows)
    print(f"\n{'=' * 74}\nerror analysis: {run['run_name']}\n{'=' * 74}")
    print(f"{total} questions\n")

    for name, description in BUCKETS:
        count = len(classified[name])
        share = count / total if total else 0.0
        print(f"  {count:3d}  {share:5.1%}  {description}")

    # Attribution: which component would have to change to fix each failure.
    retrieval = sum(
        len(classified[name])
        for name in (
            "wrong_unretrievable",
            "wrong_not_promoted",
            "wrong_evidence_gap",
            "refused_evidence_gap",
            "partial_evidence_gap",
        )
    )
    generation = sum(
        len(classified[name])
        for name in (
            "wrong_grounding",
            "refused_despite_evidence",
            "partial_despite_evidence",
            "confabulation",
        )
    )
    good = len(classified["correct"]) + len(classified["correct_refusal"])

    print(
        f"\n  {good:3d} correct, {retrieval:3d} retrieval-caused, "
        f"{generation:3d} generation-caused"
    )

    for name, description in BUCKETS:
        if name in ("correct", "correct_refusal") or not classified[name]:
            continue
        print(f"\n  {description}")
        for record in classified[name]:
            print(
                f"    {record['question_uid']:38s} {record['category']:18s} "
                f"ev {record['evidence'] if record['evidence'] is not None else 0:.2f}"
            )
            print(f"      {(record['answer'] or '')[:96]}")


if __name__ == "__main__":
    main()
