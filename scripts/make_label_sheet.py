"""Produce a blank sheet of answers for hand grading.

The judge's verdicts are not evidence until someone checks them. This writes a
stratified sample with the verdict field left empty, so the human grades
without seeing what the judge said — a sheet pre-filled with the judge's
answers measures how willing the grader is to agree, not whether the judge is
right.

    uv run python scripts/make_label_sheet.py --run-id 86582a6b --n 30
"""

import argparse
import json
from pathlib import Path

from cardterms.db import get_conn

OUTPUT_DIR = Path("data/eval/judgements")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()

    with get_conn() as conn:
        run = conn.execute(
            "SELECT id FROM runs WHERE id::text LIKE %s", (args.run_id + "%",)
        ).fetchone()
        if not run:
            raise SystemExit(f"no run matching {args.run_id!r}")

        rows = conn.execute(
            """
            SELECT q.question_uid, q.category, q.question, q.reference_answer,
                   r.answer
            FROM run_results r
            JOIN eval_questions q ON q.id = r.question_id
            WHERE r.run_id = %s
              AND q.reference_answer IS NOT NULL
              AND NOT coalesce(r.abstained, false)
            ORDER BY q.question_uid
            """,
            (run["id"],),
        ).fetchall()

    # Rotate across categories so the sample is not dominated by whichever
    # category happens to have the most answered questions.
    buckets: dict[str, list] = {}
    for row in rows:
        buckets.setdefault(row["category"], []).append(row)

    picked = []
    while len(picked) < min(args.n, len(rows)) and any(buckets.values()):
        for category in sorted(buckets):
            if buckets[category] and len(picked) < args.n:
                picked.append(buckets[category].pop(0))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / f"{str(run['id'])[:8]}_hand_labels.jsonl"
    if destination.exists():
        raise SystemExit(f"{destination} already exists — refusing to overwrite")

    lines = [
        json.dumps(
            {
                "question_uid": row["question_uid"],
                "verdict": "",
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "answer": row["answer"],
            }
        )
        for row in picked
    ]
    destination.write_text("\n".join(lines) + "\n")

    print(f"\n  {len(picked)} answers written to {destination}")
    print("  Fill in each empty verdict with: correct, partial or wrong\n")
    for category in sorted(buckets):
        count = sum(1 for r in picked if r["category"] == category)
        print(f"    {category:20s} {count}")


if __name__ == "__main__":
    main()
