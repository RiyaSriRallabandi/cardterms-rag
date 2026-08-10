"""Data quality checks over the parsed corpus.

Run after any change to ingestion. Structural checks fail the run; content
checks that depend on source document quality are reported as warnings, since
some filings legitimately lack the features being tested.

    uv run python scripts/verify_parsing.py
"""

from dataclasses import dataclass

from cardterms.db import get_conn

# A document averaging fewer characters per page than this either failed to
# extract or is a cover sheet. Corpus median is roughly 1,000.
MIN_CHARS_PER_PAGE = 150

# Page texts are joined by this separator when assembling raw_text.
PAGE_SEPARATOR_LEN = 2


@dataclass
class Result:
    name: str
    passed: bool
    detail: str
    rows: list | None = None
    fatal: bool = True


def _scalar(conn, sql: str, params=()) -> int:
    return next(iter(conn.execute(sql, params).fetchone().values()))


def check_all_parsed(conn) -> Result:
    """Every document has been parsed and produced text."""
    rows = conn.execute(
        """
        SELECT doc_uid, char_count FROM documents
        WHERE parsed_at IS NULL OR raw_text IS NULL OR char_count < 200
        ORDER BY char_count
        """
    ).fetchall()
    return Result(
        "All documents parsed with text",
        not rows,
        f"{len(rows)} document(s) unparsed or nearly empty",
        rows,
    )


def check_page_counts(conn) -> Result:
    """Parsed page count matches the page count recorded in the manifest."""
    rows = conn.execute(
        """
        SELECT d.doc_uid, d.page_count AS manifest, count(p.id) AS parsed
        FROM documents d LEFT JOIN document_pages p ON p.doc_id = d.id
        GROUP BY d.id, d.doc_uid, d.page_count
        HAVING d.page_count <> count(p.id)
        """
    ).fetchall()
    return Result(
        "Page counts match manifest",
        not rows,
        f"{len(rows)} document(s) with mismatched page counts",
        rows,
    )


def check_page_offsets_exact(conn) -> Result:
    """Text at each page's recorded offsets equals that page's stored text.

    This is what makes a chunk resolvable to a physical page; if it fails,
    every citation points somewhere wrong.
    """
    n = _scalar(
        conn,
        """
        SELECT count(*) FROM documents d JOIN document_pages p ON p.doc_id = d.id
        WHERE substring(d.raw_text FROM p.char_start + 1 FOR p.char_count) <> p.text
        """,
    )
    return Result("Page offsets resolve exactly", n == 0, f"{n} mismatched page(s)")


def check_page_offsets_contiguous(conn) -> Result:
    """Consecutive pages abut, separated only by the page separator."""
    rows = conn.execute(
        """
        SELECT d.doc_uid, p1.page_no, p1.char_end, p2.char_start
        FROM documents d
        JOIN document_pages p1 ON p1.doc_id = d.id
        JOIN document_pages p2 ON p2.doc_id = d.id AND p2.page_no = p1.page_no + 1
        WHERE p2.char_start <> p1.char_end + %s
        LIMIT 20
        """,
        (PAGE_SEPARATOR_LEN,),
    ).fetchall()
    return Result(
        "Page offsets are contiguous",
        not rows,
        f"{len(rows)} discontinuity(ies) found",
        rows,
    )


def check_table_offsets(conn) -> Result:
    """Recorded table ranges begin at a Markdown table row."""
    rows = conn.execute(
        """
        SELECT d.doc_uid, t.page_no, t.table_index
        FROM document_tables t JOIN documents d ON d.id = t.doc_id
        WHERE substring(d.raw_text FROM t.char_start + 1 FOR 1) <> '|'
        LIMIT 20
        """
    ).fetchall()
    return Result(
        "Table offsets point at table text",
        not rows,
        f"{len(rows)} table(s) with bad offsets",
        rows,
    )


def check_table_within_page(conn) -> Result:
    """Every table range falls inside the page it is attributed to."""
    rows = conn.execute(
        """
        SELECT d.doc_uid, t.page_no
        FROM document_tables t
        JOIN documents d ON d.id = t.doc_id
        JOIN document_pages p ON p.doc_id = t.doc_id AND p.page_no = t.page_no
        WHERE t.char_start < p.char_start OR t.char_end > p.char_end
        LIMIT 20
        """
    ).fetchall()
    return Result(
        "Tables fall within their page",
        not rows,
        f"{len(rows)} table(s) outside their page range",
        rows,
    )


def check_no_control_chars(conn) -> Result:
    """No form feeds or vertical tabs survive cleaning."""
    n = _scalar(
        conn,
        """
        SELECT count(*) FROM documents
        WHERE position(chr(12) in raw_text) > 0
           OR position(chr(11) in raw_text) > 0
        """,
    )
    return Result("No control characters in text", n == 0, f"{n} document(s) affected")


def check_no_mojibake(conn) -> Result:
    """Product names carry no CP437 decoding artefacts."""
    rows = conn.execute(
        """
        SELECT doc_uid, product_name FROM documents
        WHERE product_name LIKE '%┬%' OR product_name LIKE '%Ã%'
           OR filename_product LIKE '%┬%' OR filename_product LIKE '%Ã%'
        LIMIT 20
        """
    ).fetchall()
    return Result(
        "No encoding artefacts in names",
        not rows,
        f"{len(rows)} name(s) with mojibake",
        rows,
    )


def check_text_density(conn) -> Result:
    """Flag documents whose text volume is too low for their page count."""
    rows = conn.execute(
        """
        SELECT doc_uid, issuer, page_count, char_count,
               round(char_count::numeric / NULLIF(page_count, 0)) AS chars_per_page
        FROM documents
        WHERE char_count::numeric / NULLIF(page_count, 0) < %s
        ORDER BY chars_per_page
        """,
        (MIN_CHARS_PER_PAGE,),
    ).fetchall()
    return Result(
        "Text density is plausible",
        not rows,
        f"{len(rows)} document(s) below {MIN_CHARS_PER_PAGE} chars/page",
        rows,
        fatal=False,
    )


def check_expected_content(conn) -> Result:
    """Agreements should mention an interest rate somewhere."""
    rows = conn.execute(
        """
        SELECT doc_uid, issuer, char_count FROM documents
        WHERE raw_text NOT ILIKE '%annual percentage rate%'
          AND raw_text NOT ILIKE '%APR%'
          AND raw_text NOT ILIKE '%interest rate%'
        ORDER BY issuer
        """
    ).fetchall()
    return Result(
        "Documents mention interest rates",
        not rows,
        f"{len(rows)} document(s) with no rate language",
        rows,
        fatal=False,
    )


def check_tables_found(conn) -> Result:
    """Report issuers whose filings yielded no tables at all."""
    rows = conn.execute(
        """
        SELECT issuer, count(*) AS docs_without_tables
        FROM documents d
        WHERE NOT EXISTS (SELECT 1 FROM document_tables t WHERE t.doc_id = d.id)
        GROUP BY issuer ORDER BY docs_without_tables DESC
        """
    ).fetchall()
    total = sum(r["docs_without_tables"] for r in rows)
    return Result(
        "Tables detected across issuers",
        total == 0,
        f"{total} document(s) with no detected tables",
        rows,
        fatal=False,
    )


def check_text_is_readable(conn) -> Result:
    """Text layers with damaged font encodings extract as glyph codes."""
    words = (" the ", " and ", " you ", " to ", " of ", " your ", " we ", " or ")
    rows = conn.execute(
        "SELECT doc_uid, issuer, char_count, raw_text FROM documents"
    ).fetchall()
    bad = []
    for row in rows:
        text = row["raw_text"] or ""
        if len(text) < 500:
            continue
        lowered = " " + text.lower() + " "
        score = sum(lowered.count(w) for w in words) / (len(text) / 1000)
        if score < 5.0:
            bad.append(
                {
                    "doc_uid": row["doc_uid"],
                    "issuer": row["issuer"],
                    "score": round(score, 1),
                }
            )
    return Result(
        "Extracted text is readable",
        not bad,
        f"{len(bad)} document(s) with unusable text layers",
        bad,
    )


CHECKS = [
    check_all_parsed,
    check_page_counts,
    check_page_offsets_exact,
    check_page_offsets_contiguous,
    check_table_offsets,
    check_table_within_page,
    check_no_control_chars,
    check_no_mojibake,
    check_text_density,
    check_expected_content,
    check_tables_found,
    check_text_is_readable,
]


def main() -> None:
    failures = 0
    warnings = 0

    with get_conn() as conn:
        results = [check(conn) for check in CHECKS]

        summary = conn.execute(
            """
            SELECT count(*) AS docs, sum(page_count) AS pages,
                   sum(char_count) AS chars, sum(table_count) AS tables,
                   sum(ocr_page_count) AS ocr_pages
            FROM documents
            """
        ).fetchone()

    print("=" * 72)
    print("PARSING VERIFICATION")
    print("=" * 72)
    print(
        f"{summary['docs']} documents | {summary['pages']} pages | "
        f"{summary['chars']:,} chars | {summary['tables']} tables | "
        f"{summary['ocr_pages']} OCR pages\n"
    )

    for result in results:
        if result.passed:
            status = "PASS"
        elif result.fatal:
            status = "FAIL"
            failures += 1
        else:
            status = "WARN"
            warnings += 1

        print(f"[{status}] {result.name}")
        if not result.passed:
            print(f"        {result.detail}")
            for row in (result.rows or [])[:10]:
                print(f"        - {dict(row)}")

    print()
    if failures:
        print(f"{failures} structural check(s) failed. Do not freeze the text.")
        raise SystemExit(1)
    if warnings:
        print(f"All structural checks passed. {warnings} warning(s) to review.")
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
