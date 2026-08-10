"""Heading detection for structure-aware chunking.

Credit card agreements use a small number of recurring section headings.
Splitting on them places boundaries where the document itself changes topic,
and gives each chunk a section label for citation.
"""

import re

# A heading is short, carries no terminal punctuation, and is either upper case
# or title case. Requiring a letter excludes rule lines and stray numbers.
MAX_HEADING_CHARS = 80
MIN_HEADING_CHARS = 3

# Sections shorter than this are merged forward. Heading detection is
# heuristic, and capitalised clause blocks in legal text produce runs of
# false positives; merging bounds their effect on chunk size.
MIN_SECTION_CHARS = 1024

# Merged sections retain each contributing heading, forming a path. Beyond a
# few levels the label stops being useful in a citation.
MAX_PATH_PARTS = 3

HEADING_RE = re.compile(
    r"^(?P<text>(?=.*[A-Za-z])[A-Z0-9][A-Za-z0-9 ,'’&/():\-.]{2,79})$"
)

TERMINAL_PUNCT = (".", ";", ":", ",")

# Recurring headings in this corpus. A line matching one of these is treated as
# a heading even if its casing is irregular.
KNOWN_HEADINGS = {
    "annual percentage rates",
    "arbitration",
    "billing rights",
    "credit reporting",
    "definitions",
    "fees",
    "how we calculate your balance",
    "interest charges",
    "interest rates and interest charges",
    "payments",
    "pricing information",
    "rates and fees table",
    "rate and fee summary",
    "your account",
    "your billing rights",
}


def _is_upper_or_title(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if all(c.isupper() for c in letters):
        return True
    words = [w for w in text.split() if w and w[0].isalpha()]
    return bool(words) and sum(w[0].isupper() for w in words) / len(words) >= 0.7


def is_heading(line: str) -> bool:
    stripped = line.strip()
    if not (MIN_HEADING_CHARS <= len(stripped) <= MAX_HEADING_CHARS):
        return False
    if stripped.startswith("|"):
        return False  # table row
    lowered = stripped.lower().rstrip(":")
    if lowered in KNOWN_HEADINGS:
        return True
    if sum(1 for w in stripped.split() if w[:1].isalpha()) < 2:
        return False  # 'N/A', 'Y .tn', stray table cells
    if stripped.endswith(TERMINAL_PUNCT):
        return False
    if not HEADING_RE.match(stripped):
        return False
    return _is_upper_or_title(stripped)


def _is_caps_run(lines: list[str], index: int) -> bool:
    """True if neighbouring lines are also fully capitalised.

    Credit agreements set whole clauses in capitals. Those wrap into runs of
    short unpunctuated lines, each of which looks like a heading in isolation.
    """

    def all_caps(text: str) -> bool:
        letters = [c for c in text if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)

    if not all_caps(lines[index].strip()):
        return False
    neighbours = [
        lines[i].strip()
        for i in (index - 1, index + 1)
        if 0 <= i < len(lines) and lines[i].strip()
    ]
    return any(all_caps(n) for n in neighbours)


def find_sections(text: str, min_chars: int = MIN_SECTION_CHARS):
    """Return [(char_start, char_end, heading), ...] covering the whole text.

    Text before the first heading is labelled with an empty heading rather
    than discarded. Sections shorter than min_chars are merged into the
    previous section, keeping that section's heading.
    """
    lines = text.splitlines(keepends=True)
    boundaries: list[tuple[int, str]] = []
    offset = 0
    for index, line in enumerate(lines):
        if is_heading(line) and not _is_caps_run(lines, index):
            boundaries.append((offset, line.strip()))
        offset += len(line)

    if not boundaries:
        return [(0, len(text), "")]

    sections: list[tuple[int, int, str]] = []
    if boundaries[0][0] > 0:
        sections.append((0, boundaries[0][0], ""))
    for index, (start, heading) in enumerate(boundaries):
        end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(text)
        sections.append((start, end, heading))

    merged: list[tuple[int, int, list[str]]] = []
    for start, end, heading in sections:
        if merged and (end - start) < min_chars:
            prev_start, _, prev_headings = merged[-1]
            merged[-1] = (prev_start, end, prev_headings + [heading])
        else:
            merged.append((start, end, [heading]))

    # The leading section has no predecessor to merge into, so fold it forward.
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_chars:
        start, _, headings = merged.pop(0)
        _next_start, next_end, next_headings = merged[0]
        merged[0] = (start, next_end, headings + next_headings)

    return [
        (start, end, " > ".join([h for h in headings if h][:MAX_PATH_PARTS]))
        for start, end, headings in merged
    ]
