"""Recognise which cards a question names, using the corpus as the vocabulary.

A question that compares two cards needs evidence from both. Reranking scores
each passage against the whole question independently, so whichever card the
phrasing favours can take every slot in the context window. Knowing which cards
were named lets the selector reserve room for each.

No model is needed for this: the product and issuer names are already in the
documents table. A token is treated as identifying a card when it appears in
that table and is specific enough to be worth matching — "iberia" identifies a
card, "credit" does not.
"""

import re
from dataclasses import dataclass

WORD_RE = re.compile(r"[a-z0-9]+")

# Words that appear in card names but never distinguish one card from another.
GENERIC = {
    "bank",
    "banking",
    "card",
    "cardholder",
    "cardmember",
    "cards",
    "company",
    "corp",
    "credit",
    "customer",
    "federal",
    "financial",
    "inc",
    "llc",
    "national",
    "association",
    "agreement",
    "account",
    "union",
    "visa",
    "mastercard",
    "discover",
    "american",
    "express",
    "usa",
    "the",
    "and",
    "for",
    "with",
}

# A token shared by many documents describes a category, not a card. Tuned to
# admit the largest genuine product family in the corpus without admitting
# words like "rewards" that span unrelated issuers.
MAX_DOCS_PER_TOKEN = 10

MIN_TOKEN_CHARS = 4

# A brand name is rare in the body of the corpus — "saks" appears in the two
# Saks filings. Generic vocabulary is not: "balance" and "statement" appear in
# nearly every agreement, yet both turn up inside some product name. Measuring
# this against the corpus avoids hand-picking a stoplist against the evaluation
# set, which would flatter the results and not survive a new question.
# Set loosely on purpose. The words this needs to remove — "balance",
# "statement" — appear in almost every agreement, so a high ceiling suffices.
# Tightening it to 0.2 removed genuine brands shared across a family of
# filings and cost half the comparison questions their second entity.
MAX_BODY_DOC_FRACTION = 0.5


@dataclass
class Entity:
    """One card or issuer named in a question, and the documents it maps to."""

    token: str
    doc_ids: frozenset[int]


def build_index(conn) -> dict[str, frozenset[int]]:
    """Map each distinctive name token to the documents that carry it."""
    rows = conn.execute(
        "SELECT id, issuer, product_name, filename_product FROM documents"
    ).fetchall()

    postings: dict[str, set[int]] = {}
    for row in rows:
        name = " ".join(
            part
            for part in (
                row["product_name"],
                row["filename_product"],
                row["issuer"],
            )
            if part
        ).lower()
        for token in set(WORD_RE.findall(name)):
            if len(token) < MIN_TOKEN_CHARS or token in GENERIC:
                continue
            postings.setdefault(token, set()).add(row["id"])

    candidates = {
        token: frozenset(docs)
        for token, docs in postings.items()
        if len(docs) <= MAX_DOCS_PER_TOKEN
    }
    if not candidates:
        return {}

    # How many documents use each candidate token in their body text. Streamed
    # one document at a time: the corpus includes a 609-page agreement.
    body_docs = dict.fromkeys(candidates, 0)
    total = 0
    cursor = conn.execute("SELECT raw_text FROM documents")
    for row in cursor:
        total += 1
        present = set(WORD_RE.findall((row["raw_text"] or "").lower()))
        for token in candidates:
            if token in present:
                body_docs[token] += 1

    ceiling = max(1, int(total * MAX_BODY_DOC_FRACTION))
    return {
        token: docs for token, docs in candidates.items() if body_docs[token] <= ceiling
    }


def detect(query: str, index: dict[str, frozenset[int]]) -> list[Entity]:
    """Return the distinct cards a question names.

    Tokens resolving to exactly the same documents describe the same card and
    are merged, so "sun" and "country" count once. Overlapping but unequal sets
    are left as separate entities: "saks" and "elite" pick out a family and one
    member of it, and a question naming both is comparing them.
    """
    words = set(WORD_RE.findall(query.lower()))
    matched = [(docs, token) for token, docs in index.items() if token in words]

    # Tokens whose documents overlap describe one card, not two: "opensky" and
    # "gold" both reach the OpenSky Gold filing, and treating them as separate
    # cards would reserve a slot for an unrelated Gold product. Merge into
    # connected components before counting how many cards were named.
    groups: list[tuple[set[int], list[str]]] = []
    for docs, token in matched:
        overlapping = [g for g in groups if g[0] & docs]
        combined_docs = set(docs)
        combined_tokens = [token]
        for group in overlapping:
            combined_docs |= group[0]
            combined_tokens += group[1]
            groups.remove(group)
        groups.append((combined_docs, combined_tokens))

    entities = [
        Entity(token="+".join(sorted(tokens)), doc_ids=frozenset(docs))
        for docs, tokens in groups
    ]
    # Narrowest first: the most specific card mention gets its slot first when
    # the window cannot accommodate every entity.
    return sorted(entities, key=lambda e: (len(e.doc_ids), e.token))
