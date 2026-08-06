"""Build the corpus manifest from the CFPB quarterly agreement archive.

Selects documents according to configs/corpus.yaml, extracts them to
data/corpus/raw/, and records a manifest of content hashes and document
properties at data/corpus_manifest.csv.

The manifest is committed to version control in place of the PDFs; the corpus
is reproduced from it by scripts/fetch_corpus.py.
"""

import hashlib
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import yaml
from tqdm import tqdm

from cardterms.logging import configure_logging, log

ZIP_PATH = Path("data/corpus/_bulk/2026_Q1.zip")
RAW_DIR = Path("data/corpus/raw")
MANIFEST = Path("data/corpus_manifest.csv")
CORPUS_CFG = Path("configs/corpus.yaml")

# A document averaging fewer than this many characters per page contains
# scanned images rather than extractable text, and requires OCR.
SCANNED_CHAR_THRESHOLD = 100

# CFPB filenames carry a submission id: "AGREEMENT.pdf-269929.pdf".
FILE_ID_RE = re.compile(r"-(\d+)\.pdf$", re.IGNORECASE)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def issuer_from_path(zip_path: str) -> str:
    """Archive entries are organised as 'ISSUER NAME/filename.pdf'."""
    parts = zip_path.split("/")
    return parts[0] if len(parts) > 1 else "unknown"


def file_id_from_path(zip_path: str) -> str:
    """Submission id, which distinguishes filings sharing a base filename."""
    match = FILE_ID_RE.search(zip_path)
    return match.group(1) if match else "0"


def product_from_path(zip_path: str) -> str:
    """Provisional product name derived from the filename.

    Several issuers name filings with internal codes rather than card names.
    Product names are resolved from document text during parsing.
    """
    name = Path(zip_path).name
    name = FILE_ID_RE.sub("", name)
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    return re.sub(r"[_\-]+", " ", name).strip()


def profile_pdf(path: Path) -> tuple[int, bool, int]:
    """Return (page_count, is_scanned, total_chars)."""
    try:
        with fitz.open(path) as doc:
            pages = doc.page_count
            chars = sum(len(page.get_text()) for page in doc)
    except Exception as exc: # noqa: BLE001 - a malformed file must not halt a batch scan
        log.warning("pdf_unreadable", file=str(path), error=str(exc))
        return 0, False, 0
    chars_per_page = chars / pages if pages else 0
    return pages, chars_per_page < SCANNED_CHAR_THRESHOLD, chars


def select_documents(
    archive: zipfile.ZipFile,
    cfg: dict,
    bucket_of: dict[str, str],
) -> list[tuple[str, str]]:
    """Choose filings to include. Returns [(zip_path, sha256), ...].

    Within each issuer, filings are considered largest first, since longer
    documents are full agreements rather than single-page rate disclosures.
    Filings whose contents duplicate an already-selected file are skipped;
    several issuers submit the same document repeatedly within a quarter.
    """
    candidates: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for info in archive.infolist():
        if not info.filename.lower().endswith(".pdf"):
            continue
        issuer = issuer_from_path(info.filename)
        if issuer in bucket_of:
            candidates[issuer].append((info.filename, info.file_size))

    overrides = cfg.get("docs_per_issuer") or {}
    default_cap = cfg.get("max_docs_per_issuer", 5)

    selected: list[tuple[str, str]] = []
    for issuer, filings in sorted(candidates.items()):
        cap = overrides.get(issuer, default_cap)
        filings.sort(key=lambda item: -item[1])

        seen_hashes: set[str] = set()
        duplicates = 0
        for filename, _size in filings:
            if len(seen_hashes) >= cap:
                break
            digest = hashlib.sha256(archive.read(filename)).hexdigest()
            if digest in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(digest)
            selected.append((filename, digest))

        log.info(
            "issuer_selected",
            issuer=issuer,
            kept=len(seen_hashes),
            available=len(filings),
            duplicates_skipped=duplicates,
        )

    return selected


def apply_exclusions(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop filings that cannot support per-document attribution."""
    before = len(df)

    max_pages = cfg.get("max_pages_per_doc")
    if max_pages:
        oversize = df.page_count > max_pages
        for row in df[oversize].itertuples():
            log.info(
                "excluded_consolidated_filing",
                doc_uid=row.doc_uid,
                pages=row.page_count,
            )
        df = df[~oversize]

    for pattern in cfg.get("exclude_patterns") or []:
        matched = df.doc_uid.str.contains(pattern, case=False, regex=False)
        for row in df[matched].itertuples():
            log.info("excluded_by_pattern", doc_uid=row.doc_uid, pattern=pattern)
        df = df[~matched]

    if len(df) != before:
        log.info("exclusions_applied", before=before, after=len(df))

    return df


def sync_raw_dir(keep: set[str]) -> None:
    """Remove extracted files not present in the manifest."""
    for pdf in RAW_DIR.glob("*.pdf"):
        if pdf.stem not in keep:
            pdf.unlink()
            log.info("removed_excluded_file", doc_uid=pdf.stem)


def print_profile(df: pd.DataFrame) -> None:
    print("\n" + "=" * 62)
    print("CORPUS PROFILE")
    print("=" * 62)
    print(f"Documents        : {len(df)}")
    print(f"Issuers          : {df.issuer.nunique()}")
    print(f"Total pages      : {df.page_count.sum()}")
    print(f"Median pages/doc : {df.page_count.median():.0f}")
    print(
        f"Scanned docs     : {df.is_scanned.sum()} ({100 * df.is_scanned.mean():.1f}%)"
    )
    print(f"Est. tokens      : {df.total_chars.sum() / 4:,.0f}")

    print("\nBy segment:")
    print(
        df.groupby("bucket").agg(
            docs=("doc_uid", "count"),
            issuers=("issuer", "nunique"),
            pages=("page_count", "sum"),
        )
    )

    print("\nIssuers by document count:")
    print(df.groupby("issuer").size().sort_values(ascending=False).head(15))

    if df.is_scanned.any():
        print("\nScanned documents requiring OCR:")
        print(df[df.is_scanned][["doc_uid", "page_count"]].to_string(index=False))


def main() -> None:
    configure_logging(json_output=False)
    cfg = yaml.safe_load(CORPUS_CFG.read_text())

    bucket_of: dict[str, str] = {}
    for bucket, names in cfg["issuers"].items():
        for name in names or []:
            bucket_of[name] = bucket

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as archive:
        selected = select_documents(archive, cfg, bucket_of)

        found = {issuer_from_path(path) for path, _ in selected}
        missing = set(bucket_of) - found
        if missing:
            log.warning("issuers_not_found_in_archive", issuers=sorted(missing))

        rows = []
        for zip_name, digest in tqdm(selected, desc="Extracting"):
            issuer = issuer_from_path(zip_name)
            product = product_from_path(zip_name)
            doc_uid = (
                f"{slugify(issuer)}__{slugify(product)}__{file_id_from_path(zip_name)}"
            )

            out_path = RAW_DIR / f"{doc_uid}.pdf"
            if not out_path.exists():
                out_path.write_bytes(archive.read(zip_name))

            pages, is_scanned, chars = profile_pdf(out_path)
            rows.append(
                {
                    "doc_uid": doc_uid,
                    "issuer": issuer,
                    "bucket": bucket_of[issuer],
                    "product_name": product,
                    "zip_path": zip_name,
                    "quarter": cfg["quarter"],
                    "sha256": digest,
                    "size_bytes": out_path.stat().st_size,
                    "page_count": pages,
                    "total_chars": chars,
                    "is_scanned": is_scanned,
                }
            )

    df = pd.DataFrame(rows).sort_values(["bucket", "issuer", "product_name"])
    df = apply_exclusions(df, cfg)

    assert df.doc_uid.is_unique, "doc_uid collision detected"

    sync_raw_dir(set(df.doc_uid))

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, index=False)

    print_profile(df)
    log.info("manifest_written", path=str(MANIFEST), rows=len(df))


if __name__ == "__main__":
    main()
