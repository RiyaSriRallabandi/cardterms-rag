"""Inspect which cards the entity detector finds in each evaluation question.

Entity-aware selection can only reserve a slot for a card it recognises. This
prints what the detector sees, and whether the documents a question actually
needs fall inside the entities it detected — a question whose gold documents
are unreachable through any detected entity cannot be helped by the selector,
however well the rest of the pipeline works.

    uv run python scripts/check_entities.py
    uv run python scripts/check_entities.py --category comparison
"""

import argparse

from cardterms.db import get_conn
from cardterms.eval.relevance import relevant_documents
from cardterms.retrieve.entities import build_index, detect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", help="restrict to one question category")
    args = parser.parse_args()

    with get_conn() as conn:
        index = build_index(conn)
        gold = relevant_documents(conn)
        rows = conn.execute(
            "SELECT id, question_uid, category, question "
            "FROM eval_questions ORDER BY category, question_uid"
        ).fetchall()

    print(f"\nvocabulary: {len(index)} distinctive tokens\n")

    multi = 0
    reachable = 0
    needs_multi = 0
    # Questions about one card that the detector splits into several: these are
    # where entity-aware selection can do harm rather than good.
    multi_by_category: dict[str, list[int]] = {}

    for row in rows:
        if args.category and row["category"] != args.category:
            continue

        entities = detect(row["question"], index)
        need = gold.get(row["id"], set())
        union = set().union(*(e.doc_ids for e in entities)) if entities else set()

        bucket = multi_by_category.setdefault(row["category"], [0, 0])
        bucket[1] += 1
        if len(entities) >= 2:
            multi += 1
            bucket[0] += 1
        if len(need) > 1:
            needs_multi += 1
            if need <= union:
                reachable += 1

        covered = "-" if not need else f"{len(need & union)}/{len(need)}"
        print(f"  {row['question_uid']:38s} {row['category']:18s} gold {covered}")
        print(f"    {row['question'][:96]}")
        if entities:
            for entity in entities:
                hits = len(entity.doc_ids & need)
                mark = "*" if hits else " "
                print(
                    f"     {mark} {entity.token:28s} {len(entity.doc_ids):3d} docs"
                    f"   {hits} gold"
                )
        else:
            print("       (no card recognised)")

    print("\n  questions resolving to two or more entities, by category")
    for category in sorted(multi_by_category):
        hit, total = multi_by_category[category]
        print(f"    {category:20s} {hit:3d}/{total:3d}")

    print(
        f"\n  {multi} questions name two or more cards\n"
        f"  {reachable}/{needs_multi} multi-document questions have every gold "
        f"document inside a detected entity"
    )


if __name__ == "__main__":
    main()
