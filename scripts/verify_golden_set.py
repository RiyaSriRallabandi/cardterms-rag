"""Checks over the golden set."""

from cardterms.db import get_conn

EXPECTED = {
    "single_fact": 20,
    "entity_confusable": 20,
    "table_lookup": 10,
    "comparison": 10,
    "unanswerable": 10,
    "ambiguous": 5,
}

NO_LABEL_CATEGORIES = {"unanswerable", "ambiguous"}


def main() -> None:
    failures = 0
    with get_conn() as conn:
        counts = {
            row["category"]: row["n"]
            for row in conn.execute(
                "SELECT category, count(*) AS n FROM eval_questions GROUP BY category"
            ).fetchall()
        }
        total = sum(counts.values())

        print("=" * 62)
        print("GOLDEN SET VERIFICATION")
        print("=" * 62)
        print(f"{total} questions\n")
        for category, expected in EXPECTED.items():
            got = counts.get(category, 0)
            flag = "ok " if got >= expected * 0.8 else "LOW"
            print(f"  [{flag}] {category:20s} {got:3d}  (target {expected})")

        unlabelled = conn.execute(
            """
            SELECT q.question_uid, q.category FROM eval_questions q
            LEFT JOIN eval_labels l ON l.question_id = q.id
            WHERE l.id IS NULL AND q.category <> ALL(%s)
            """,
            (list(NO_LABEL_CATEGORIES),),
        ).fetchall()

        wrongly_labelled = conn.execute(
            """
            SELECT DISTINCT q.question_uid FROM eval_questions q
            JOIN eval_labels l ON l.question_id = q.id
            WHERE q.category = ANY(%s)
            """,
            (list(NO_LABEL_CATEGORIES),),
        ).fetchall()

        bad_spans = conn.execute(
            """
            SELECT count(*) AS n FROM eval_labels l JOIN documents d ON d.id = l.doc_id
            WHERE substring(d.raw_text FROM l.char_start + 1
                            FOR l.char_end - l.char_start) <> l.quote
            """
        ).fetchone()["n"]

        long_quotes = conn.execute(
            "SELECT count(*) AS n FROM eval_labels WHERE char_end - char_start > 1200"
        ).fetchone()["n"]

        multi = conn.execute(
            """
            SELECT count(*) AS n FROM (
                SELECT question_id FROM eval_labels
                GROUP BY question_id HAVING count(*) > 1
            ) t
            """
        ).fetchone()["n"]

        print()
        for name, bad, detail in [
            ("Answerable questions have labels", unlabelled, "questions with no label"),
            (
                "Unanswerable questions have none",
                wrongly_labelled,
                "questions labelled that should not be",
            ),
        ]:
            ok = not bad
            failures += 0 if ok else 1
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
            if bad:
                print(f"        {len(bad)} {detail}")
                for row in bad[:5]:
                    print(f"        - {row['question_uid']}")

        ok = bad_spans == 0
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] Label spans resolve to their quote")
        if bad_spans:
            print(f"        {bad_spans} span(s) do not match")

        print(f"[{'ok ' if long_quotes == 0 else 'WARN'}] Quotes are minimal")
        if long_quotes:
            print(f"        {long_quotes} quote(s) exceed 1200 characters")

        print(f"[info] {multi} question(s) carry more than one label")

    print()
    if failures:
        raise SystemExit(f"{failures} check(s) failed.")
    print("All checks passed.")


if __name__ == "__main__":
    main()
