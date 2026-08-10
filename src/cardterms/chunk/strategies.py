"""Chunking strategies.

All strategies split a document into atomic spans and pack them to a token
budget. Detected tables are single atoms and are never split: separating a fee
value from its row label makes the value unusable.

Chunks are produced as character spans into documents.raw_text, which lets a
chunk resolve to a page and lets span-based evaluation labels be matched to
chunks regardless of which strategy produced them.
"""

import re
from dataclasses import dataclass

from cardterms.chunk.sections import find_sections
from cardterms.chunk.tokenizer import count_tokens

SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")

# Characters per token, used only to size hard splits before exact counting.
CHARS_PER_TOKEN = 4

# Fixed-window atoms are a fraction of the chunk budget. Atoms as large as the
# budget leave nothing to pack or carry forward, which disables overlap.
FIXED_ATOM_DIVISOR = 8


@dataclass
class Span:
    start: int
    end: int
    is_table: bool = False
    section: str = ""


@dataclass
class Chunk:
    char_start: int
    char_end: int
    token_count: int
    section: str
    is_table: bool
    is_parent: bool = False
    parent_index: int | None = None


def _split_on(text: str, base: int, pattern: re.Pattern) -> list[tuple[int, int]]:
    """Split a span on a regex, returning absolute character spans."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            spans.append((base + cursor, base + match.start()))
        cursor = match.end()
    if cursor < len(text):
        spans.append((base + cursor, base + len(text)))
    return [(s, e) for s, e in spans if e > s]


def _hard_split(start: int, end: int, max_tokens: int) -> list[tuple[int, int]]:
    """Split a span that has no usable internal boundary."""
    width = max(1, max_tokens * CHARS_PER_TOKEN)
    return [(s, min(s + width, end)) for s in range(start, end, width)]


def _prose_atoms(text: str, start: int, end: int, max_tokens: int) -> list[Span]:
    """Paragraphs, subdivided into sentences and then hard splits if oversized."""
    atoms: list[Span] = []
    for para_start, para_end in _split_on(text[start:end], start, PARAGRAPH_SPLIT_RE):
        if count_tokens(text[para_start:para_end]) <= max_tokens:
            atoms.append(Span(para_start, para_end))
            continue

        sentences = _split_on(text[para_start:para_end], para_start, SENTENCE_END_RE)
        for sent_start, sent_end in sentences or [(para_start, para_end)]:
            if count_tokens(text[sent_start:sent_end]) <= max_tokens:
                atoms.append(Span(sent_start, sent_end))
            else:
                atoms += [
                    Span(s, e) for s, e in _hard_split(sent_start, sent_end, max_tokens)
                ]
    return atoms


def _atoms_in_range(
    text: str,
    start: int,
    end: int,
    table_spans: list[tuple[int, int]],
    max_tokens: int,
) -> list[Span]:
    """Atoms for a character range.

    Tables are emitted whole and excluded from prose splitting, so their text
    is never counted twice.
    """
    atoms: list[Span] = []
    cursor = start

    for table_start, table_end in sorted(table_spans):
        if table_end <= start or table_start >= end:
            continue
        table_start = max(table_start, start)
        table_end = min(table_end, end)
        if table_start > cursor:
            atoms += _prose_atoms(text, cursor, table_start, max_tokens)
        atoms.append(Span(table_start, table_end, is_table=True))
        cursor = table_end

    if cursor < end:
        atoms += _prose_atoms(text, cursor, end, max_tokens)

    return atoms


def _atoms_fixed(
    text: str, table_spans: list[tuple[int, int]], max_tokens: int
) -> list[Span]:
    """Fixed windows over prose, tables still atomic. The naive baseline."""
    window = max(1, max_tokens // FIXED_ATOM_DIVISOR)
    atoms: list[Span] = []
    cursor = 0

    for table_start, table_end in sorted(table_spans):
        if table_start > cursor:
            atoms += [Span(s, e) for s, e in _hard_split(cursor, table_start, window)]
        atoms.append(Span(table_start, table_end, is_table=True))
        cursor = table_end

    if cursor < len(text):
        atoms += [Span(s, e) for s, e in _hard_split(cursor, len(text), window)]

    return atoms


def _make_chunk(spans: list[Span], token_count: int) -> Chunk:
    return Chunk(
        char_start=spans[0].start,
        char_end=spans[-1].end,
        token_count=token_count,
        section=spans[0].section,
        is_table=all(span.is_table for span in spans),
    )


def _pack(
    text: str, atoms: list[Span], max_tokens: int, overlap_pct: float
) -> list[Chunk]:
    """Combine atoms into chunks up to the token budget."""
    chunks: list[Chunk] = []
    current: list[Span] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        chunks.append(_make_chunk(current, current_tokens))

        if overlap_pct <= 0:
            current, current_tokens = [], 0
            return

        budget = int(max_tokens * overlap_pct)
        carried: list[Span] = []
        carried_tokens = 0
        for span in reversed(current):
            span_tokens = count_tokens(text[span.start : span.end])
            if carried_tokens + span_tokens > budget:
                break
            carried.insert(0, span)
            carried_tokens += span_tokens
        current, current_tokens = carried, carried_tokens

    for atom in atoms:
        atom_tokens = count_tokens(text[atom.start : atom.end])
        if current and current_tokens + atom_tokens > max_tokens:
            flush()
        current.append(atom)
        current_tokens += atom_tokens

    if current:
        chunks.append(_make_chunk(current, current_tokens))

    return chunks


def chunk_document(
    text: str,
    table_spans: list[tuple[int, int]],
    strategy: str,
    max_tokens: int,
    overlap_pct: float = 0.0,
) -> list[Chunk]:
    if strategy == "fixed":
        atoms = _atoms_fixed(text, table_spans, max_tokens)
        return _pack(text, atoms, max_tokens, overlap_pct)

    if strategy == "recursive":
        atoms = _atoms_in_range(text, 0, len(text), table_spans, max_tokens)
        return _pack(text, atoms, max_tokens, overlap_pct)

    if strategy == "structure_aware":
        chunks: list[Chunk] = []
        for sec_start, sec_end, heading in find_sections(text):
            atoms = _atoms_in_range(text, sec_start, sec_end, table_spans, max_tokens)
            for atom in atoms:
                atom.section = heading
            for chunk in _pack(text, atoms, max_tokens, overlap_pct):
                chunk.section = heading
                chunks.append(chunk)
        return chunks

    if strategy == "parent_doc":
        parents = chunk_document(
            text, table_spans, "structure_aware", max_tokens * 2, 0.0
        )
        out: list[Chunk] = []
        for parent in parents:
            parent.is_parent = True
            out.append(parent)
        for index, parent in enumerate(parents):
            atoms = _atoms_in_range(
                text, parent.char_start, parent.char_end, table_spans, max_tokens
            )
            for child in _pack(text, atoms, max_tokens, overlap_pct):
                child.section = parent.section
                child.parent_index = index
                out.append(child)
        return out

    raise ValueError(f"unknown strategy: {strategy}")
