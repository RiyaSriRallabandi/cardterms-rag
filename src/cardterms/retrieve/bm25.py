"""Keyword retrieval over a chunk set.

BM25 ranks by term overlap, weighting rare terms more heavily and normalising
for chunk length. It requires no model and no embeddings, which makes it both
the natural first baseline and the sparse half of hybrid retrieval later.

Tokenisation keeps decimals and percentages intact, since '35.99%' and '$41.00'
are the terms cardholder questions actually turn on.
"""

import re

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*%?")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunk_ids: list[int], texts: list[str], doc_ids: list[int]):
        self.chunk_ids = chunk_ids
        self.doc_ids = doc_ids
        self.index = BM25Okapi([tokenize(text) for text in texts])

    @classmethod
    def from_chunk_set(cls, conn, chunk_set_name: str) -> "BM25Retriever":
        rows = conn.execute(
            """
            SELECT c.id, c.text, c.doc_id
            FROM chunks c JOIN chunk_sets s ON s.id = c.chunk_set_id
            WHERE s.name = %s AND NOT c.is_parent
            ORDER BY c.id
            """,
            (chunk_set_name,),
        ).fetchall()
        if not rows:
            raise ValueError(f"chunk set {chunk_set_name!r} is empty or missing")
        return cls(
            [row["id"] for row in rows],
            [row["text"] for row in rows],
            [row["doc_id"] for row in rows],
        )

    def search(self, query: str, top_k: int) -> list[tuple[int, int, float]]:
        """Return [(chunk_id, doc_id, score), ...] ranked best first."""
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.chunk_ids[i], self.doc_ids[i], float(scores[i])) for i in ranked]
