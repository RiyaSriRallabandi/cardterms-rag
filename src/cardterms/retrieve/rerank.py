"""Cross-encoder reranking of retrieval candidates.

Retrievers are bi-encoders: query and passage are encoded independently, so a
passage is represented the same way regardless of what was asked. A
cross-encoder reads both together in a single forward pass, which is markedly
more accurate and far too slow to apply to a whole corpus — so it is applied to
a shortlist produced by a cheaper retriever.
"""

from functools import lru_cache

BATCH_SIZE = 32

MODELS = {
    "ms-marco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge": "BAAI/bge-reranker-base",
}


@lru_cache(maxsize=1)
def load(hf_name: str):
    import torch
    from sentence_transformers import CrossEncoder

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return CrossEncoder(hf_name, device=device)


class CrossEncoderReranker:
    def __init__(self, conn, model_key: str, augment: bool = False):
        if model_key not in MODELS:
            raise ValueError(f"unknown reranker: {model_key}")
        self.conn = conn
        self.model_key = model_key
        self.augment = augment
        self.model = load(MODELS[model_key])

    def _texts(self, chunk_ids: list[int]) -> dict[int, str]:
        """Chunk text, optionally prefixed with the product it belongs to.

        Many chunks are bare table fragments naming no product, so a
        cross-encoder scoring them against a question that names a card has
        nothing to match on. The retriever already identifies the correct
        document 90% of the time; supplying that identity lets the reranker
        use it.
        """
        rows = self.conn.execute(
            """
            SELECT c.id, c.text, c.section_path,
                   d.issuer, d.product_name, d.filename_product
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE c.id = ANY(%s)
            """,
            (chunk_ids,),
        ).fetchall()

        texts = {}
        for row in rows:
            if self.augment:
                product = row["product_name"] or row["filename_product"] or ""
                section = row["section_path"] or ""
                header = f"{product} — {row['issuer']}"
                if section:
                    header += f" — {section}"
                texts[row["id"]] = f"{header}\n{row['text']}"
            else:
                texts[row["id"]] = row["text"]
        return texts

    def rerank(
        self, query: str, candidates: list[tuple[int, int, float]], top_n: int
    ) -> list[tuple[int, int, float]]:
        """Reorder candidates by cross-encoder score, keeping the best top_n."""
        if not candidates:
            return []

        texts = self._texts([chunk_id for chunk_id, _, _ in candidates])
        pairs = [(query, texts.get(chunk_id, "")) for chunk_id, _, _ in candidates]
        scores = self.model.predict(
            pairs, batch_size=BATCH_SIZE, show_progress_bar=False
        )

        ranked = sorted(zip(candidates, scores, strict=True), key=lambda item: -item[1])
        return [
            (chunk_id, doc_id, float(score))
            for (chunk_id, doc_id, _), score in ranked[:top_n]
        ]
