"""Find documents that answer a question but are not yet labelled.

Several issuers file one agreement under multiple brand names, so a question
may be answered correctly by more than one document. Labelling only one would
score a correct retrieval as a miss.

Matches are candidates, not conclusions: identical boilerplate in a different
product's agreement does not answer a question about a specific card.
"""

from cardterms.db import get_conn

# Quotes shorter than this match too broadly to be informative.
MIN_QUOTE_CHARS = 40


def main() -> None:
    with get_conn() as conn:
        labels = conn.execute(
            """
            SELECT l.quote, l.doc_id, q.id AS question_id,
                   q.question_uid, q.question, q.category
            FROM eval_labels l JOIN eval_questions q ON q.id = l.question_id
            ORDER BY q.question_uid
            """
        ).fetchall()
        documents = conn.execute(
            "SELECT id, doc_uid, raw_text FROM documents"
        ).fetchall()
        labelled = {
            (row["question_id"], row["doc_id"])
            for row in conn.execute(
                "SELECT question_id, doc_id FROM eval_labels"
            ).fetchall()
        }

    short = 0
    found = 0

    for label in labels:
        quote = label["quote"].strip()
        if len(quote) < MIN_QUOTE_CHARS:
            short += 1
            continue

        others = [
            doc["doc_uid"]
            for doc in documents
            if (label["question_id"], doc["id"]) not in labelled
            and quote in doc["raw_text"]
        ]
        if not others:
            continue

        found += 1
        print(f"\n{label['question_uid']}  [{label['category']}]")
        print(f"  {label['question']}")
        print(f"  quote: {quote[:70]}...")
        print(f"  also appears verbatim in {len(others)} unlabelled document(s):")
        for doc_uid in sorted(others)[:6]:
            print(f"    - {doc_uid}")

    print(f"\n{found} label(s) with candidates elsewhere.")
    print(f"{short} quote(s) skipped as too short to match meaningfully.")


if __name__ == "__main__":
    main()
