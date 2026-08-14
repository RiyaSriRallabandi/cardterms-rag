"""Grade a run's answers with an LLM judge.

The mechanical checks ask whether the figure from the reference answer appears
somewhere in the generated text. That cannot separate "the annual fee is $59"
from "the annual fee is not $59" or "$59 is the cash advance fee", so the
accuracy it reports is an upper bound. A judge reads both answers and decides
whether they say the same thing.

A judge is only worth its verdicts if it agrees with a human. Its output is
written to a file so it can be compared against hand labels rather than
trusted; see scripts/calibrate_judge.py.

    uv run python scripts/judge_run.py --run-id 86582a6b
"""

import argparse
import json
from pathlib import Path

from cardterms.db import get_conn
from cardterms.eval.llm import complete_json
from cardterms.logging import configure_logging, log

PROMPT_DIR = Path("prompts")
OUTPUT_DIR = Path("data/eval/judgements")
VERDICTS = ("correct", "partial", "wrong")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="run id prefix")
    parser.add_argument("--prompt", default="judge_v1")
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    args = parser.parse_args()

    configure_logging(json_output=False)
    template = (PROMPT_DIR / f"{args.prompt}.txt").read_text()

    with get_conn() as conn:
        run = conn.execute(
            "SELECT id, run_name FROM runs WHERE id::text LIKE %s",
            (args.run_id + "%",),
        ).fetchone()
        if not run:
            raise SystemExit(f"no run matching {args.run_id!r}")

        rows = conn.execute(
            """
            SELECT q.question_uid, q.category, q.question, q.reference_answer,
                   r.answer, r.abstained
            FROM run_results r
            JOIN eval_questions q ON q.id = r.question_id
            WHERE r.run_id = %s
              AND q.reference_answer IS NOT NULL
              AND NOT coalesce(r.abstained, false)
            ORDER BY q.question_uid
            """,
            (run["id"],),
        ).fetchall()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / f"{str(run['id'])[:8]}_{args.prompt}.jsonl"

    graded = []
    for index, row in enumerate(rows, start=1):
        prompt = template.format(
            question=row["question"],
            reference=row["reference_answer"],
            answer=row["answer"],
        )
        try:
            result = complete_json(
                prompt, provider=args.provider, model=args.model, temperature=0.0
            )
            verdict = str(result.get("verdict", "")).strip().lower()
            reason = str(result.get("reason", ""))[:200]
        except Exception as error:  # noqa: BLE001 - one bad grade must not stop the run
            log.warning("judge_failed", question=row["question_uid"], error=str(error))
            verdict, reason = "", "judge error"

        if verdict not in VERDICTS:
            log.warning("judge_unparsed", question=row["question_uid"], verdict=verdict)
            verdict = "unparsed"

        graded.append(
            {
                "question_uid": row["question_uid"],
                "category": row["category"],
                "verdict": verdict,
                "reason": reason,
            }
        )
        print(f"  [{index:3d}/{len(rows)}] {row['question_uid']:38s} {verdict}")

    destination.write_text("\n".join(json.dumps(g) for g in graded) + "\n")

    counts = {v: sum(1 for g in graded if g["verdict"] == v) for v in VERDICTS}
    unparsed = sum(1 for g in graded if g["verdict"] not in VERDICTS)

    print(f"\n  {run['run_name']}  ({str(run['id'])[:8]})")
    print(f"  graded {len(graded)} answered questions -> {destination}\n")
    for verdict in VERDICTS:
        share = counts[verdict] / len(graded) if graded else 0.0
        print(f"    {verdict:10s} {counts[verdict]:3d}   {share:.3f}")
    if unparsed:
        print(f"    unparsed   {unparsed:3d}")


if __name__ == "__main__":
    main()
