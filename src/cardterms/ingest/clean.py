"""Text normalisation and removal of repeated page furniture."""

import re
import unicodedata
from collections import Counter

# Repeated-line detection is unreliable on very short documents, where a line
# may legitimately appear on most pages.
MIN_PAGES_FOR_BOILERPLATE = 4

# Proportion of pages a line must appear on to be treated as page furniture.
BOILERPLATE_PAGE_RATIO = 0.7

PAGE_NUMBER_RE = re.compile(r"(?:page\s*)?\d{1,3}(?:\s*of\s*\d{1,3})?", re.IGNORECASE)
HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
HORIZONTAL_SPACE_RE = re.compile(r"[ \t]+")
BLANK_RUN_RE = re.compile(r"\n{3,}")

# Control characters carry no meaning in extracted text and cannot be stored
# in a text column. Tab, newline and carriage return are preserved.
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def find_boilerplate_lines(page_texts: list[str]) -> set[str]:
    """Lines recurring across most pages, i.e. running headers and footers."""
    if len(page_texts) < MIN_PAGES_FOR_BOILERPLATE:
        return set()

    counts: Counter[str] = Counter()
    for text in page_texts:
        unique_lines = {line.strip() for line in text.splitlines() if line.strip()}
        for line in unique_lines:
            if 3 <= len(line) <= 120:
                counts[line] += 1

    threshold = BOILERPLATE_PAGE_RATIO * len(page_texts)
    return {line for line, count in counts.items() if count >= threshold}


def clean_page(text: str, boilerplate: set[str]) -> str:
    """Normalise a page's text.

    Case is preserved deliberately: issuer and product names are the highest
    value tokens for keyword retrieval, and lowercasing removes that signal.
    """
    text = CONTROL_CHAR_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped in boilerplate:
            continue
        if stripped and PAGE_NUMBER_RE.fullmatch(stripped):
            continue
        lines.append(HORIZONTAL_SPACE_RE.sub(" ", stripped))

    return BLANK_RUN_RE.sub("\n\n", "\n".join(lines)).strip()
