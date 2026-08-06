"""Reproduce the exact corpus from the committed manifest.

Downloads the CFPB quarterly archive, extracts only the documents listed in
data/corpus_manifest.csv, and verifies each against its recorded SHA-256.
"""

import hashlib
import zipfile
from pathlib import Path

import httpx
import pandas as pd
import yaml
from tqdm import tqdm

from cardterms.logging import configure_logging, log

MANIFEST = Path("data/corpus_manifest.csv")
CORPUS_CFG = Path("configs/corpus.yaml")
BULK_DIR = Path("data/corpus/_bulk")
RAW_DIR = Path("data/corpus/raw")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    if dest.exists():
        log.info("archive_already_present", path=str(dest))
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading_archive", url=url)
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(dest, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as bar,
        ):
            for chunk in r.iter_bytes(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))


def main() -> None:
    configure_logging(json_output=False)
    cfg = yaml.safe_load(CORPUS_CFG.read_text())
    df = pd.read_csv(MANIFEST)

    zip_path = BULK_DIR / f"{cfg['quarter'].replace('-', '_')}.zip"
    download(cfg["bulk_zip_url"], zip_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok = mismatched = 0

    with zipfile.ZipFile(zip_path) as z:
        for row in tqdm(df.itertuples(), total=len(df), desc="Verifying"):
            out = RAW_DIR / f"{row.doc_uid}.pdf"
            if not out.exists():
                out.write_bytes(z.read(row.zip_path))
            if sha256_of(out) == row.sha256:
                ok += 1
            else:
                mismatched += 1
                log.error("hash_mismatch", doc_uid=row.doc_uid)

    log.info("corpus_verified", verified=ok, mismatched=mismatched, total=len(df))
    if mismatched:
        raise SystemExit(f"{mismatched} file(s) failed verification")
    print(f"\n✅ All {ok} documents verified against manifest.")


if __name__ == "__main__":
    main()
