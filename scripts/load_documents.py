"""Load manifest rows into the documents table. Idempotent."""

from pathlib import Path

import pandas as pd
import yaml

from cardterms.db import get_conn
from cardterms.logging import configure_logging, log

MANIFEST = Path("data/corpus_manifest.csv")
CORPUS_CFG = Path("configs/corpus.yaml")


def main() -> None:
    configure_logging(json_output=False)
    cfg = yaml.safe_load(CORPUS_CFG.read_text())
    source_url = cfg["bulk_zip_url"]
    df = pd.read_csv(MANIFEST)

    with get_conn() as conn:
        for row in df.itertuples():
            conn.execute(
                """
                INSERT INTO documents
                    (doc_uid, issuer, product_name, filename_product,
                     effective_quarter, source_url, sha256, page_count,
                     is_scanned)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_uid) DO UPDATE SET
                    issuer            = EXCLUDED.issuer,
                    product_name      = CASE
                                          WHEN documents.parsed_at IS NULL
                                          THEN EXCLUDED.product_name
                                          ELSE documents.product_name
                                        END,
                    filename_product  = EXCLUDED.filename_product,
                    effective_quarter = EXCLUDED.effective_quarter,
                    sha256            = EXCLUDED.sha256,
                    page_count        = EXCLUDED.page_count,
                    is_scanned        = EXCLUDED.is_scanned
                """,
                (
                    row.doc_uid,
                    row.issuer,
                    row.product_name,
                    row.product_name,
                    row.quarter,
                    source_url,
                    row.sha256,
                    int(row.page_count),
                    bool(row.is_scanned),
                ),
            )

        # Documents dropped from the manifest are removed so that the table
        # mirrors the manifest exactly. Pages and tables cascade.
        removed = conn.execute(
            "DELETE FROM documents WHERE doc_uid <> ALL(%s)",
            (list(df.doc_uid),),
        ).rowcount

        conn.commit()

    with get_conn() as conn:
        n = conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"]

    log.info("documents_loaded", count=n, manifest_rows=len(df), removed=removed)
    print(f"\n✅ {n} documents in database (manifest has {len(df)} rows).")
    if removed:
        print(f"   {removed} document(s) removed — no longer in manifest.")


if __name__ == "__main__":
    main()
