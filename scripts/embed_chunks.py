"""Embed a chunk set with a given model.

Work already done is skipped, so re-running is cheap and a failed run resumes
where it stopped. Chunks are ordered by length before batching so that padding
within a batch is minimal, which matters on a memory-constrained machine.
"""

import argparse

from pgvector.psycopg import register_vector
from tqdm import tqdm

from cardterms.db import get_conn
from cardterms.embed.models import REGISTRY, encode_passages
from cardterms.logging import configure_logging, log

WRITE_BATCH = 256


def table_for(dimensions: int) -> str:
    return f"embeddings_{dimensions}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--chunk-set", required=True)
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="omit instruction prefixes, for the prefix ablation",
    )
    args = parser.parse_args()

    configure_logging(json_output=False)
    model = REGISTRY[args.model]
    scheme = "none" if args.no_prefix else "standard"
    table = table_for(model.dimensions)

    with get_conn() as conn:
        register_vector(conn)

        pending = conn.execute(
            f"""
            SELECT c.id, c.text, c.token_count
            FROM chunks c JOIN chunk_sets s ON s.id = c.chunk_set_id
            WHERE s.name = %s
              AND NOT EXISTS (
                  SELECT 1 FROM {table} e
                  WHERE e.chunk_id = c.id AND e.model = %s
                    AND e.prefix_scheme = %s
              )
            ORDER BY c.token_count
            """,
            (args.chunk_set, model.key, scheme),
        ).fetchall()

    if not pending:
        print("nothing to do — already embedded")
        return

    truncated = sum(1 for row in pending if row["token_count"] > model.max_tokens)
    if truncated:
        log.warning(
            "chunks_exceed_model_limit",
            model=model.key,
            limit=model.max_tokens,
            affected=truncated,
            total=len(pending),
        )

    log.info(
        "embedding_start",
        model=model.key,
        chunk_set=args.chunk_set,
        prefix_scheme=scheme,
        chunks=len(pending),
    )

    written = 0
    insert_sql = f"""
        INSERT INTO {table} (chunk_id, model, prefix_scheme, vec)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """

    # One connection for the run, and one round trip per batch rather than one
    # per row: inserts dominated the previous implementation's wall time.
    with get_conn() as conn:
        register_vector(conn)
        for start in tqdm(range(0, len(pending), WRITE_BATCH), desc=model.key):
            batch = pending[start : start + WRITE_BATCH]
            vectors = encode_passages(
                model, [row["text"] for row in batch], use_prefix=not args.no_prefix
            )
            with conn.cursor() as cur:
                cur.executemany(
                    insert_sql,
                    [
                        (row["id"], model.key, scheme, vector)
                        for row, vector in zip(batch, vectors, strict=True)
                    ],
                )
            conn.commit()
            written += len(batch)

    log.info("embedding_complete", model=model.key, written=written)
    print(f"\n{written} chunks embedded with {model.key} ({scheme} prefixes)")
    if truncated:
        print(
            f"  {truncated} chunk(s) exceeded the model's {model.max_tokens}-token "
            f"limit and were truncated"
        )


if __name__ == "__main__":
    main()
