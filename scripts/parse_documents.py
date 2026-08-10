"""Parse corpus PDFs into cleaned text with page and table offsets.

Writes documents.raw_text together with document_pages and document_tables,
which give every character in a document a page attribution and mark the
regions that must not be split during chunking.
"""

import argparse
import re
from pathlib import Path

from tqdm import tqdm

from cardterms.db import get_conn
from cardterms.ingest.clean import clean_page, find_boilerplate_lines
from cardterms.ingest.parse import parse_pdf
from cardterms.ingest.product import extract_product_name
from cardterms.logging import configure_logging, log

RAW_DIR = Path("data/corpus/raw")
PAGE_SEPARATOR = "\n\n"
PARSER_NAME = "pymupdf+tesseract"

# Length of the leading fragment used to locate a table within cleaned text.
TABLE_ANCHOR_CHARS = 80


def assemble(
    cleaned_pages: list[str], parsed_pages, boilerplate: set[str]
) -> tuple[str, list, list, int]:
    """Join pages into one text and record page and table character offsets."""
    parts: list[str] = []
    page_rows: list[dict] = []
    table_rows: list[dict] = []
    unlocated = 0
    offset = 0

    for cleaned, parsed in zip(cleaned_pages, parsed_pages, strict=True):
        start = offset
        parts.append(cleaned)
        offset += len(cleaned)

        page_rows.append(
            {
                "page_no": parsed.page_no,
                "char_start": start,
                "char_end": offset,
                "text": cleaned,
                "char_count": len(cleaned),
                "ocr_applied": parsed.ocr_applied,
                "ocr_regions": parsed.ocr_regions,
            }
        )

        for table in parsed.tables:
            # Tables are rendered before cleaning, so the same normalisation is
            # applied to the search string before locating it in cleaned text.
            normalised = clean_page(table.markdown, boilerplate)
            local = cleaned.find(normalised[:TABLE_ANCHOR_CHARS])
            if local < 0:
                unlocated += 1
                continue

            table_rows.append(
                {
                    "page_no": table.page_no,
                    "table_index": table.index,
                    "char_start": start + local,
                    "char_end": start + local + len(normalised),
                    "n_rows": table.n_rows,
                    "n_cols": table.n_cols,
                    "empty_cells": table.empty_cells,
                    "ocr_cells_filled": table.ocr_cells_filled,
                }
            )

        parts.append(PAGE_SEPARATOR)
        offset += len(PAGE_SEPARATOR)

    return "".join(parts), page_rows, table_rows, unlocated


def parse_document(doc, extract_tables: bool) -> dict:
    parsed = parse_pdf(RAW_DIR / f"{doc['doc_uid']}.pdf", extract_tables)

    raw_pages = [page.text for page in parsed.pages]
    boilerplate = find_boilerplate_lines(raw_pages)
    cleaned_pages = [clean_page(text, boilerplate) for text in raw_pages]

    raw_text, page_rows, table_rows, unlocated = assemble(
        cleaned_pages, parsed.pages, boilerplate
    )
    product_name, source = extract_product_name(raw_text)

    return {
        "raw_text": raw_text,
        "page_rows": page_rows,
        "table_rows": table_rows,
        "unlocated_tables": unlocated,
        "product_name": product_name or doc["filename_product"] or doc["product_name"],
        "product_name_source": source,
        "ocr_page_count": parsed.ocr_page_count,
        "ocr_region_count": sum(page.ocr_regions for page in parsed.pages),
        "table_count": parsed.table_count,
    }


def persist(conn, doc_id: int, result: dict) -> None:
    conn.execute("DELETE FROM document_pages WHERE doc_id = %s", (doc_id,))
    conn.execute("DELETE FROM document_tables WHERE doc_id = %s", (doc_id,))

    conn.execute(
        """
        UPDATE documents SET
            raw_text            = %s,
            char_count          = %s,
            product_name        = %s,
            product_name_source = %s,
            ocr_page_count      = %s,
            table_count         = %s,
            parser              = %s,
            parsed_at           = now()
        WHERE id = %s
        """,
        (
            result["raw_text"],
            len(result["raw_text"]),
            result["product_name"],
            result["product_name_source"],
            result["ocr_page_count"],
            result["table_count"],
            PARSER_NAME,
            doc_id,
        ),
    )

    for row in result["page_rows"]:
        conn.execute(
            """
            INSERT INTO document_pages
                (doc_id, page_no, char_start, char_end, text, char_count,
                 ocr_applied, ocr_regions)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                doc_id,
                row["page_no"],
                row["char_start"],
                row["char_end"],
                row["text"],
                row["char_count"],
                row["ocr_applied"],
                row["ocr_regions"],
            ),
        )

    for row in result["table_rows"]:
        conn.execute(
            """
            INSERT INTO document_tables
                (doc_id, page_no, table_index, char_start, char_end,
                 n_rows, n_cols, empty_cells, ocr_cells_filled)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                doc_id,
                row["page_no"],
                row["table_index"],
                row["char_start"],
                row["char_end"],
                row["n_rows"],
                row["n_cols"],
                row["empty_cells"],
                row["ocr_cells_filled"],
            ),
        )


# Chase identifies filings by an internal collection code. One filing entity
# submits them under descriptive filenames and the other under the bare code,
# so a filing with no product name can take the name of a sibling filing
# sharing its code.
COLLECTION_CODE_RE = re.compile(r"col(\d{4,6})", re.IGNORECASE)


def _strip_code(name: str) -> str:
    return COLLECTION_CODE_RE.sub("", name or "").strip(" _-")


def backfill_shared_collection_names(conn) -> int:
    """Name filings that carry only a collection code, using sibling filings."""
    rows = conn.execute(
        "SELECT id, doc_uid, product_name_source, filename_product FROM documents"
    ).fetchall()

    descriptive: dict[str, str] = {}
    for row in rows:
        match = COLLECTION_CODE_RE.search(row["doc_uid"])
        if not match:
            continue
        candidate = _strip_code(row["filename_product"])
        if len(candidate) > 4 and match.group(1) not in descriptive:
            descriptive[match.group(1)] = candidate

    updated = 0
    for row in rows:
        if row["product_name_source"] != "filename":
            continue
        match = COLLECTION_CODE_RE.search(row["doc_uid"])
        if not match:
            continue
        if len(_strip_code(row["filename_product"])) > 4:
            continue  # this filing already names itself
        name = descriptive.get(match.group(1))
        if not name:
            continue
        conn.execute(
            "UPDATE documents SET product_name = %s, product_name_source = %s "
            "WHERE id = %s",
            (name.title(), "shared_collection_code", row["id"]),
        )
        updated += 1

    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tables", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--like", help="only parse doc_uids containing this substring")
    args = parser.parse_args()

    configure_logging(json_output=False)

    with get_conn() as conn:
        query = "SELECT id, doc_uid, product_name, filename_product FROM documents"
        params: list = []
        if args.like:
            query += " WHERE doc_uid LIKE %s"
            params.append(f"%{args.like}%")
        query += " ORDER BY doc_uid"
        if args.limit:
            query += f" LIMIT {int(args.limit)}"
        documents = conn.execute(query, params).fetchall()

    failures = 0
    backfilled = 0
    totals = {"ocr_pages": 0, "ocr_regions": 0, "tables": 0, "unlocated": 0}

    for doc in tqdm(documents, desc="Parsing"):
        try:
            result = parse_document(doc, extract_tables=not args.no_tables)
        except Exception as exc:  # noqa: BLE001 - one bad file must not halt the run
            log.error("parse_failed", doc_uid=doc["doc_uid"], error=str(exc))
            failures += 1
            continue

        totals["ocr_pages"] += result["ocr_page_count"]
        totals["ocr_regions"] += result["ocr_region_count"]
        totals["tables"] += result["table_count"]
        totals["unlocated"] += result["unlocated_tables"]

        with get_conn() as conn:
            persist(conn, doc["id"], result)
            conn.commit()

    with get_conn() as conn:
        backfilled = backfill_shared_collection_names(conn)
        conn.commit()
    if backfilled:
        log.info("collection_code_backfill", updated=backfilled)

    log.info("parsing_complete", documents=len(documents), failures=failures, **totals)

    print(f"\nDocuments parsed   : {len(documents) - failures}/{len(documents)}")
    print(f"Pages recovered OCR : {totals['ocr_pages']}")
    print(f"Image regions OCR   : {totals['ocr_regions']}")
    print(f"Tables extracted    : {totals['tables']}")
    if totals["unlocated"]:
        print(f"Tables not located  : {totals['unlocated']}  (offsets not recorded)")
    if backfilled:
        print(f"Names from siblings : {backfilled}")


if __name__ == "__main__":
    main()
