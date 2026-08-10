"""Build a chunk set over the parsed corpus.

Chunk sets coexist so that chunking strategies can be compared without
re-ingesting. Each set records the configuration that produced it.

    uv run python scripts/build_chunks.py --strategy recursive --tokens 512 --overlap 0.15
"""

import argparse
import hashlib
import json

from tqdm import tqdm

from cardterms.chunk.strategies import chunk_document
from cardterms.chunk.tokenizer import models_that_fit
from cardterms.db import get_conn
from cardterms.logging import configure_logging, log


def set_name(strategy: str, tokens: int, overlap: float) -> str:
    return f"{strategy}_{tokens}_ov{int(overlap * 100)}"


def page_for(pages: list[dict], char_start: int, char_end: int) -> tuple[int, int]:
    """Page numbers spanned by a character range."""
    touched = [
        p["page_no"]
        for p in pages
        if p["char_start"] < char_end and p["char_end"] > char_start
    ]
    return (min(touched), max(touched)) if touched else (0, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["fixed", "recursive", "structure_aware", "parent_doc"],
    )
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--overlap", type=float, default=0.0)
    args = parser.parse_args()

    configure_logging(json_output=False)
    name = set_name(args.strategy, args.tokens, args.overlap)
    config = {
        "strategy": args.strategy,
        "max_tokens": args.tokens,
        "overlap_pct": args.overlap,
    }

    with get_conn() as conn:
        conn.execute("DELETE FROM chunk_sets WHERE name = %s", (name,))
        row = conn.execute(
            "INSERT INTO chunk_sets (name, config) VALUES (%s, %s) RETURNING id",
            (name, json.dumps(config)),
        ).fetchone()
        chunk_set_id = row["id"]

        documents = conn.execute(
            "SELECT id, doc_uid, raw_text FROM documents ORDER BY doc_uid"
        ).fetchall()
        conn.commit()

    total = 0
    oversized = 0

    for doc in tqdm(documents, desc=name):
        with get_conn() as conn:
            pages = conn.execute(
                "SELECT page_no, char_start, char_end FROM document_pages "
                "WHERE doc_id = %s ORDER BY page_no",
                (doc["id"],),
            ).fetchall()
            tables = conn.execute(
                "SELECT char_start, char_end FROM document_tables "
                "WHERE doc_id = %s ORDER BY char_start",
                (doc["id"],),
            ).fetchall()

        table_spans = [(t["char_start"], t["char_end"]) for t in tables]
        chunks = chunk_document(
            doc["raw_text"], table_spans, args.strategy, args.tokens, args.overlap
        )

        with get_conn() as conn:
            parent_row_ids: list[int] = []
            for index, chunk in enumerate(chunks):
                text = doc["raw_text"][chunk.char_start : chunk.char_end]
                if not text.strip():
                    continue
                if chunk.token_count > args.tokens and not chunk.is_table:
                    oversized += 1

                page_start, page_end = page_for(pages, chunk.char_start, chunk.char_end)
                inserted = conn.execute(
                    """
                    INSERT INTO chunks
                        (chunk_set_id, doc_id, chunk_index, text, token_count,
                         page_start, page_end, char_start, char_end,
                         section_path, is_table, is_parent, parent_chunk_id,
                         content_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        chunk_set_id,
                        doc["id"],
                        index,
                        text,
                        chunk.token_count,
                        page_start,
                        page_end,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.section or None,
                        chunk.is_table,
                        chunk.is_parent,
                        parent_row_ids[chunk.parent_index]
                        if chunk.parent_index is not None
                        else None,
                        hashlib.sha256(text.encode()).hexdigest(),
                    ),
                ).fetchone()
                if chunk.is_parent:
                    parent_row_ids.append(inserted["id"])
                total += 1
            conn.commit()

    with get_conn() as conn:
        conn.execute(
            "UPDATE chunk_sets SET n_chunks = %s WHERE id = %s", (total, chunk_set_id)
        )
        conn.commit()

    log.info(
        "chunk_set_built",
        name=name,
        chunks=total,
        oversized=oversized,
        usable_by=models_that_fit(args.tokens),
    )
    print(f"\n{name}: {total} chunks")
    if oversized:
        print(f"  {oversized} exceed the budget (indivisible atoms)")
    print(
        f"  models able to consume without truncation: {models_that_fit(args.tokens)}"
    )


if __name__ == "__main__":
    main()
