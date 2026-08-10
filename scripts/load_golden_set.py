"""Load the verified golden set, resolving quotes to character spans.

Quotes are located in documents.raw_text and stored as character offsets, so a
label is independent of any chunking strategy.

A quote that cannot be located, or that appears more than once in its document,
is a hard error. Both cases would otherwise produce a label pointing at text
that does not answer the question, which counts as an answer no system can
retrieve and silently depresses every recall figure.
"""

import json
from pathlib import Path

from cardterms.db import get_conn
from cardterms.logging import configure_logging, log

GOLDEN_PATH = Path("data/eval/golden_set.jsonl")


def page_for(pages: list[dict], char_start: int) -> int | None:
    for page in pages:
        if page["char_start"] <= char_start < page["char_end"]:
            return page["page_no"]
    return None


def main() -> None:
    configure_logging(json_output=False)

    entries = [
        json.loads(line)
        for line in GOLDEN_PATH.read_text().splitlines()
        if line.strip()
    ]
    entries = [e for e in entries if e.get("keep", True)]

    with get_conn() as conn:
        docs = {
            row["doc_uid"]: row
            for row in conn.execute(
                "SELECT id, doc_uid, raw_text FROM documents"
            ).fetchall()
        }
        pages_by_doc: dict[int, list[dict]] = {}
        for row in conn.execute(
            "SELECT doc_id, page_no, char_start, char_end FROM document_pages "
            "ORDER BY doc_id, page_no"
        ).fetchall():
            pages_by_doc.setdefault(row["doc_id"], []).append(dict(row))

        conn.execute("DELETE FROM eval_questions")

        failures: list[str] = []
        n_labels = 0

        for entry in entries:
            row = conn.execute(
                """
                INSERT INTO eval_questions
                    (question_uid, question, category, source,
                     reference_answer, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    entry["question_uid"],
                    entry["question"],
                    entry["category"],
                    entry.get("source", "llm_drafted_human_verified"),
                    entry.get("reference_answer"),
                    entry.get("notes"),
                ),
            ).fetchone()
            question_id = row["id"]

            for label in entry.get("labels", []):
                doc = docs.get(label["doc_uid"])
                if doc is None:
                    failures.append(
                        f"{entry['question_uid']}: unknown doc {label['doc_uid']}"
                    )
                    continue

                quote = label["quote"]

                positions: list[int] = []
                cursor = doc["raw_text"].find(quote)
                while cursor >= 0:
                    positions.append(cursor)
                    cursor = doc["raw_text"].find(quote, cursor + 1)

                if not positions:
                    failures.append(
                        f"{entry['question_uid']}: quote not found in "
                        f"{label['doc_uid']}"
                    )
                    continue

                # Issuers restate key terms in both a summary table and the
                # agreement body. Where that is deliberate, every occurrence is
                # a valid answer and all are labelled; otherwise an ambiguous
                # quote is an error, since one arbitrary match would be wrong.
                if len(positions) > 1 and not label.get("all_occurrences"):
                    failures.append(
                        f"{entry['question_uid']}: quote appears "
                        f"{len(positions)} times in {label['doc_uid']} — "
                        f"not uniquely locatable"
                    )
                    continue

                for position in positions:
                    conn.execute(
                        """
                        INSERT INTO eval_labels
                            (question_id, doc_id, page, char_start, char_end,
                             quote, relevance)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (
                            question_id,
                            doc["id"],
                            page_for(pages_by_doc.get(doc["id"], []), position),
                            position,
                            position + len(quote),
                            quote,
                            label.get("relevance", 2),
                        ),
                    )
                    n_labels += 1

        if failures:
            conn.rollback()
            print(f"\n{len(failures)} label(s) failed:\n")
            for failure in failures:
                print("  ", failure)
            raise SystemExit("Nothing was loaded.")

        conn.commit()

    log.info("golden_set_loaded", questions=len(entries), labels=n_labels)
    print(f"\n{len(entries)} questions, {n_labels} labels loaded.")


if __name__ == "__main__":
    main()
