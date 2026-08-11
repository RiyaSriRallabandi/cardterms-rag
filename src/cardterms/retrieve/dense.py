"""Vector search over an embedded chunk set.

Distance uses the cosine operator. Vectors are normalised, so inner product
would rank identically and slightly faster, but cosine is chosen because it
stays correct if a future model is added without normalisation — a silent
class of error otherwise.
"""

from pgvector.psycopg import register_vector

from cardterms.embed.models import REGISTRY, encode_query


class DenseRetriever:
    def __init__(self, conn, chunk_set: str, model_key: str, use_prefix: bool = True):
        self.conn = conn
        self.chunk_set = chunk_set
        self.model = REGISTRY[model_key]
        self.use_prefix = use_prefix
        self.scheme = "standard" if use_prefix else "none"
        self.table = f"embeddings_{self.model.dimensions}"
        register_vector(conn)

        count = conn.execute(
            f"""
            SELECT count(*) AS n FROM {self.table} e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN chunk_sets s ON s.id = c.chunk_set_id
            WHERE s.name = %s AND e.model = %s AND e.prefix_scheme = %s
            """,
            (chunk_set, self.model.key, self.scheme),
        ).fetchone()["n"]
        if count == 0:
            raise ValueError(
                f"no embeddings for {model_key} on {chunk_set} ({self.scheme})"
            )

    def search(self, query: str, top_k: int) -> list[tuple[int, int, float]]:
        vector = encode_query(self.model, query, use_prefix=self.use_prefix)
        rows = self.conn.execute(
            f"""
            SELECT c.id AS chunk_id, c.doc_id, 1 - (e.vec <=> %s) AS score
            FROM {self.table} e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN chunk_sets s ON s.id = c.chunk_set_id
            WHERE s.name = %s AND e.model = %s AND e.prefix_scheme = %s
              AND NOT c.is_parent
            ORDER BY e.vec <=> %s
            LIMIT %s
            """,
            (vector, self.chunk_set, self.model.key, self.scheme, vector, top_k),
        ).fetchall()
        return [(r["chunk_id"], r["doc_id"], float(r["score"])) for r in rows]
