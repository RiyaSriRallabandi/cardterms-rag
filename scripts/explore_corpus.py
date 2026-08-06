"""Survey the CFPB bulk archive.

Usage:
    uv run python scripts/explore_corpus.py            # full survey
    uv run python scripts/explore_corpus.py CAPITAL    # search issuer names
"""

import sys
import zipfile
from collections import Counter
from pathlib import Path

ZIP_PATH = Path("data/corpus/_bulk/2026_Q1.zip")


def load_issuer_counts() -> Counter:
    with zipfile.ZipFile(ZIP_PATH) as z:
        pdfs = [n for n in z.namelist() if n.lower().endswith(".pdf")]
    return Counter(n.split("/")[0] for n in pdfs if "/" in n)


def search(needle: str) -> None:
    counts = load_issuer_counts()
    needle = needle.upper()
    hits = [(f, c) for f, c in counts.items() if needle in f.upper()]
    print(f"\n--- Issuers matching '{needle}' ({len(hits)} found) ---")
    for folder, count in sorted(hits, key=lambda t: -t[1]):
        print(f"  {count:5d}  {folder}")
    if not hits:
        print("  (none)")


def survey() -> None:
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = z.namelist()
    pdfs = [n for n in names if n.lower().endswith(".pdf")]

    print(f"Total entries : {len(names)}")
    print(f"PDF files     : {len(pdfs)}")
    print(f"Directories   : {len(names) - len(pdfs)}")

    print("\n--- First 15 PDF paths ---")
    for n in pdfs[:15]:
        print(" ", n)

    counts = Counter(n.split("/")[0] for n in pdfs if "/" in n)
    print(f"\nDistinct issuers: {len(counts)}")
    print("--- 40 largest by document count ---")
    for folder, count in counts.most_common(40):
        print(f"  {count:5d}  {folder}")


def main() -> None:
    if len(sys.argv) > 1:
        search(sys.argv[1])
    else:
        survey()


if __name__ == "__main__":
    main()
