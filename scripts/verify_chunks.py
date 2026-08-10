"""Structural checks over chunk sets."""

from collections import defaultdict

from cardterms.db import get_conn


def uncovered_characters(conn, chunk_set_id: int) -> int:
    """Non-whitespace characters covered by no chunk.

    Chunk ranges overlap when overlap is configured, so coverage is computed by
    merging intervals per document rather than by comparing adjacent rows.
    """
    rows = conn.execute(
        """
        SELECT doc_id, char_start, char_end FROM chunks
        WHERE chunk_set_id = %s AND NOT is_parent
        ORDER BY doc_id, char_start
        """,
        (chunk_set_id,),
    ).fetchall()

    spans: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        spans[row["doc_id"]].append((row["char_start"], row["char_end"]))

    texts = {
        row["id"]: row["raw_text"]
        for row in conn.execute("SELECT id, raw_text FROM documents").fetchall()
    }

    lost = 0
    for doc_id, doc_spans in spans.items():
        merged: list[list[int]] = []
        for start, end in sorted(doc_spans):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        text = texts[doc_id]
        cursor = 0
        for start, end in merged:
            if start > cursor:
                lost += len(text[cursor:start].strip())
            cursor = max(cursor, end)
        if cursor < len(text):
            lost += len(text[cursor:].strip())

    return lost


def main() -> None:
    failures = 0
    with get_conn() as conn:
        sets = conn.execute(
            "SELECT id, name, n_chunks, config FROM chunk_sets ORDER BY name"
        ).fetchall()

        print("=" * 72)
        print("CHUNK SET VERIFICATION")
        print("=" * 72)

        for cs in sets:
            stats = conn.execute(
                """
                SELECT count(*) AS n,
                       round(avg(token_count)) AS mean_tokens,
                       percentile_disc(0.5) WITHIN GROUP (ORDER BY token_count) AS p50,
                       max(token_count) AS max_tokens,
                       sum((is_table)::int) AS tables,
                       count(DISTINCT doc_id) AS docs
                FROM chunks WHERE chunk_set_id = %s
                """,
                (cs["id"],),
            ).fetchone()

            bad_offsets = conn.execute(
                """
                SELECT count(*) AS n FROM chunks c JOIN documents d ON d.id = c.doc_id
                WHERE c.chunk_set_id = %s
                  AND substring(d.raw_text FROM c.char_start + 1
                                FOR c.char_end - c.char_start) <> c.text
                """,
                (cs["id"],),
            ).fetchone()["n"]

            split_tables = conn.execute(
                """
                SELECT count(*) AS n
                FROM document_tables t
                JOIN chunks c ON c.doc_id = t.doc_id AND c.chunk_set_id = %s
                WHERE (c.char_start > t.char_start AND c.char_start < t.char_end)
                   OR (c.char_end > t.char_start AND c.char_end < t.char_end)
                """,
                (cs["id"],),
            ).fetchone()["n"]

            no_page = conn.execute(
                "SELECT count(*) AS n FROM chunks "
                "WHERE chunk_set_id = %s AND page_start = 0",
                (cs["id"],),
            ).fetchone()["n"]

            # Text falling between consecutive chunks appears in no chunk and is
            # therefore unreachable by retrieval. Whitespace gaps are expected:
            # paragraph separators are consumed during splitting.
            uncovered = uncovered_characters(conn, cs["id"])

            ok = (
                bad_offsets == 0
                and split_tables == 0
                and no_page == 0
                and uncovered == 0
            )
            failures += 0 if ok else 1

            print(f"\n[{'PASS' if ok else 'FAIL'}] {cs['name']}")
            print(
                f"        {stats['n']:6d} chunks over {stats['docs']} docs | "
                f"mean {stats['mean_tokens']} tok | p50 {stats['p50']} | "
                f"max {stats['max_tokens']} | {stats['tables']} table chunks"
            )
            if bad_offsets:
                print(f"        offsets do not resolve: {bad_offsets}")
            if split_tables:
                print(f"        chunk boundaries inside tables: {split_tables}")
            if no_page:
                print(f"        chunks with no page attribution: {no_page}")
            if uncovered:
                print(f"        non-whitespace characters in gaps: {uncovered}")

    print()
    if failures:
        raise SystemExit(f"{failures} chunk set(s) failed.")
    print("All chunk sets passed.")


if __name__ == "__main__":
    main()
