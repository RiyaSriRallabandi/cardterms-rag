"""Assemble retrieved passages into prompt context.

Passages are ordered so that the highest-ranked appear at the beginning and end
rather than consecutively. Language models attend least reliably to the middle
of a long context, so burying the best passage there wastes it.

Each passage carries its product, issuer and page, which is what makes a
citation resolvable to something a person can check.
"""

from dataclasses import dataclass

from cardterms.chunk.tokenizer import count_tokens


@dataclass
class Passage:
    number: int
    chunk_id: int
    doc_id: int
    product: str
    issuer: str
    page: int
    text: str


def _interleave(items: list) -> list:
    """Place items so rank 1 is first, rank 2 last, rank 3 second, and so on."""
    ordered: list = [None] * len(items)
    low, high = 0, len(items) - 1
    for index, item in enumerate(items):
        if index % 2 == 0:
            ordered[low] = item
            low += 1
        else:
            ordered[high] = item
            high -= 1
    return ordered


def build_passages(
    conn, retrieved: list[tuple[int, int, float]], max_tokens: int
) -> list[Passage]:
    """Fetch chunk text and metadata, order for attention, and fit the budget."""
    chunk_ids = [chunk_id for chunk_id, _, _ in retrieved]
    if not chunk_ids:
        return []

    rows = {
        row["id"]: row
        for row in conn.execute(
            """
            SELECT c.id, c.text, c.page_start, c.doc_id,
                   d.issuer, d.product_name, d.filename_product
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE c.id = ANY(%s)
            """,
            (chunk_ids,),
        ).fetchall()
    }

    selected, used = [], 0
    for chunk_id, doc_id, _ in retrieved:
        row = rows.get(chunk_id)
        if row is None:
            continue
        tokens = count_tokens(row["text"])
        if used + tokens > max_tokens and selected:
            break
        selected.append((chunk_id, doc_id, row))
        used += tokens

    ordered = _interleave(selected)

    return [
        Passage(
            number=position,
            chunk_id=chunk_id,
            doc_id=doc_id,
            product=row["product_name"] or row["filename_product"] or "",
            issuer=row["issuer"],
            page=row["page_start"],
            text=row["text"],
        )
        for position, (chunk_id, doc_id, row) in enumerate(ordered, start=1)
    ]


def render(passages: list[Passage]) -> str:
    return "\n\n".join(
        f"[{p.number}] {p.product} — {p.issuer} — page {p.page}\n{p.text}"
        for p in passages
    )
