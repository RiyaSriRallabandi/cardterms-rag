"""Draft candidate evaluation questions from sampled passages.

Passages are sampled across market segments and biased towards text containing
monetary amounts and rates, which is what cardholder questions concern. Output
is a review file: every draft must be verified by hand before it becomes part
of the evaluation set.
"""

import argparse
import json
import re
import time
from pathlib import Path

from tqdm import tqdm

from cardterms.db import get_conn
from cardterms.eval.llm import complete_json
from cardterms.logging import configure_logging, log

DRAFTS_PATH = Path("data/eval/drafts.jsonl")
CHUNK_SET = "recursive_512_ov0"

SYSTEM = (
    "You write evaluation questions for a question-answering system over "
    "credit card cardholder agreements. You are precise and you never invent "
    "facts."
)

PROMPT = """Below is a passage from the cardholder agreement for: {product}
Issued by: {issuer}

PASSAGE:
\"\"\"
{passage}
\"\"\"

Write up to {n} questions that a real cardholder would ask and that THIS
PASSAGE answers directly.

Rules:
- Questions must sound like a person, not like a search query.
- Do not quote the passage in the question.
- Each question must name the specific card, so it is unambiguous.
- Only ask what the passage actually answers. If the passage answers nothing a
  cardholder would ask, return an empty list.
- The quote must be copied VERBATIM from the passage, and must be the shortest
  span that fully answers the question.
- The reference answer must be one short factual sentence.

Return JSON of the form:
{{"questions": [
   {{"question": "...", "quote": "...", "reference_answer": "...",
     "category": "single_fact" | "table_lookup"}}
]}}"""

MONEY_OR_RATE = re.compile(r"(\$\s?\d|\d+\.\d+\s?%|\d+\s?%)")


def sample_passages(limit_per_bucket: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.id AS chunk_id, c.text, c.char_start, c.char_end,
                   c.page_start, d.id AS doc_id, d.doc_uid, d.issuer,
                   d.product_name, d.filename_product, d.bucket
            FROM chunks c
            JOIN chunk_sets s ON s.id = c.chunk_set_id
            JOIN documents d ON d.id = c.doc_id
            WHERE s.name = %s AND c.token_count BETWEEN 120 AND 520
            ORDER BY random()
            """,
            (CHUNK_SET,),
        ).fetchall()

    by_bucket: dict[str, list[dict]] = {}
    seen_docs: set[int] = set()
    for row in rows:
        if not MONEY_OR_RATE.search(row["text"]):
            continue
        bucket = row["bucket"]
        chosen = by_bucket.setdefault(bucket, [])
        if len(chosen) >= limit_per_bucket:
            continue
        # At most two passages per document, for coverage across issuers.
        if sum(1 for c in chosen if c["doc_id"] == row["doc_id"]) >= 2:
            continue
        chosen.append(dict(row))
        seen_docs.add(row["doc_id"])

    return [row for rows_ in by_bucket.values() for row in rows_]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-bucket", type=int, default=14)
    parser.add_argument("--per-passage", type=int, default=2)
    args = parser.parse_args()

    configure_logging(json_output=False)
    passages = sample_passages(args.per_bucket)
    log.info("passages_sampled", count=len(passages))

    DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(DRAFTS_PATH, "w") as out:
        for passage in tqdm(passages, desc="Drafting"):
            product = passage["product_name"] or passage["filename_product"]
            try:
                result = complete_json(
                    PROMPT.format(
                        product=product,
                        issuer=passage["issuer"],
                        passage=passage["text"],
                        n=args.per_passage,
                    ),
                    system=SYSTEM,
                )
            except Exception as exc:  # noqa: BLE001 - one passage must not stop the run
                log.warning("draft_failed", doc=passage["doc_uid"], error=str(exc))
                continue

            for index, item in enumerate(result.get("questions", [])):
                quote = (item.get("quote") or "").strip()
                if not quote or quote not in passage["text"]:
                    log.debug("quote_not_verbatim", doc=passage["doc_uid"])
                    continue

                out.write(
                    json.dumps(
                        {
                            "question_uid": f"{passage['doc_uid'][:40]}_{passage['chunk_id']}_{index}",
                            "keep": True,
                            "category": item.get("category", "single_fact"),
                            "question": item.get("question", "").strip(),
                            "reference_answer": item.get(
                                "reference_answer", ""
                            ).strip(),
                            "labels": [{"doc_uid": passage["doc_uid"], "quote": quote}],
                            "notes": "",
                            "_issuer": passage["issuer"],
                            "_product": product,
                            "_page": passage["page_start"],
                            "_context": passage["text"],
                        }
                    )
                    + "\n"
                )
                written += 1
            time.sleep(5.0)  # free-tier budget is token-based, not request-based

    log.info("drafts_written", path=str(DRAFTS_PATH), count=written)
    print(f"\n{written} drafts written to {DRAFTS_PATH}")


if __name__ == "__main__":
    main()
