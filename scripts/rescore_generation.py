"""Recompute generation checks from stored answer text.

The first generation run classified refusals by exact string match, which missed
every answer where the model wrote "INSUFFICIENT CONTEXT" instead of
"INSUFFICIENT_CONTEXT". The generated text is stored, so the affected checks can
be recomputed without paying for inference again.

Citation validity is not recomputed: it needs the passage list, which is not
persisted. Take that number from a fresh run.

    uv run python scripts/rescore_generation.py
    uv run python scripts/rescore_generation.py --run-id 15930f75-...
"""

import argparse

from cardterms.db import get_conn
from cardterms.generate.answer import INSUFFICIENT_RE, NEEDS_CLARIFICATION_RE
from cardterms.generate.validate import (
    CITATION_RE,
    FIGURE_RE,
    MIN_CLAIM_CHARS,
    SENTENCE_SPLIT_RE,
)

UNSCORED_CATEGORIES = ("unanswerable", "ambiguous")


def classify(text: str) -> str | None:
    if INSUFFICIENT_RE.search(text):
        return "insufficient_context"
    if NEEDS_CLARIFICATION_RE.search(text):
        return "needs_clarification"
    return None


def uncited_claims(text: str) -> int:
    return sum(
        1
        for s in SENTENCE_SPLIT_RE.split(text)
        if len(s.strip()) >= MIN_CLAIM_CHARS and not CITATION_RE.search(s)
    )


def has_reference_figure(text: str, reference: str | None) -> bool | None:
    if not reference:
        return None
    figures = FIGURE_RE.findall(reference)
    if not figures:
        return None
    normalised = text.replace(" ", "")
    return any(f.replace(" ", "") in normalised for f in figures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", help="defaults to the most recent run")
    args = parser.parse_args()

    with get_conn() as conn:
        if args.run_id:
            run = conn.execute(
                "SELECT id, run_name FROM runs WHERE id::text LIKE %s",
                (args.run_id + "%",),
            ).fetchone()
        else:
            run = conn.execute(
                "SELECT id, run_name FROM runs WHERE run_name LIKE '%%gen-%%' "
                "ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        if not run:
            raise SystemExit("no generation run found")

        rows = conn.execute(
            """
            SELECT q.question_uid, q.category, q.reference_answer,
                   r.answer, r.abstained AS stored_abstained
            FROM run_results r
            JOIN eval_questions q ON q.id = r.question_id
            WHERE r.run_id = %s
            ORDER BY q.category, q.question_uid
            """,
            (run["id"],),
        ).fetchall()

    scored = [r for r in rows if r["category"] not in UNSCORED_CATEGORIES]

    reclassified = []
    for r in rows:
        text = r["answer"] or ""
        kind = classify(text)
        reclassified.append(
            {
                **dict(r),
                "kind": kind,
                "abstained": kind is not None,
                "changed": bool(kind is not None) != bool(r["stored_abstained"]),
            }
        )

    changed = [r for r in reclassified if r["changed"]]

    print(f"\n{'=' * 66}\nrescored: {run['run_name']}\n{'=' * 66}")
    print(f"{len(changed)} of {len(rows)} answers reclassified\n")
    for r in changed:
        print(f"  {r['question_uid']:34s} {r['category']:18s} -> abstained")

    print("\n  abstention (corrected)")
    for category in ("unanswerable", "ambiguous"):
        group = [r for r in reclassified if r["category"] == category]
        ok = sum(1 for r in group if r["abstained"])
        print(f"    {category:18s} {ok}/{len(group)}")

    false_abstain = [
        r
        for r in reclassified
        if r["category"] not in UNSCORED_CATEGORIES and r["abstained"]
    ]
    print(f"    false abstention   {len(false_abstain)}/{len(scored)}")
    for r in false_abstain:
        print(f"      {r['question_uid']:36s} {r['category']:14s} {r['kind']}")

    # Which refusal token the model reaches for, against which it should have.
    print("\n  token choice on refusals")
    for r in reclassified:
        if r["category"] in UNSCORED_CATEGORIES and r["abstained"]:
            expected = (
                "insufficient_context"
                if r["category"] == "unanswerable"
                else "needs_clarification"
            )
            mark = "ok " if r["kind"] == expected else "SWAP"
            print(f"    [{mark}] {r['question_uid']:34s} {r['kind']}")

    answered = [
        r
        for r in reclassified
        if r["category"] not in UNSCORED_CATEGORIES and not r["abstained"]
    ]
    figures = [
        (r, has_reference_figure(r["answer"] or "", r["reference_answer"]))
        for r in answered
    ]
    checked = [(r, v) for r, v in figures if v is not None]
    hits = sum(1 for _, v in checked if v)
    uncited = sum(uncited_claims(r["answer"] or "") for r in answered)

    print("\n  answered questions (corrected denominator)")
    print(f"    n                      {len(answered)}")
    print(f"    expected figure present {hits}/{len(checked)}")
    print(f"    uncited claims          {uncited} sentence(s)")

    print("\n  confabulations (should have refused, gave a figure)")
    for r in reclassified:
        if r["category"] in UNSCORED_CATEGORIES and not r["abstained"]:
            print(f"    {r['question_uid']:34s} {(r['answer'] or '')[:70]}")


if __name__ == "__main__":
    main()
