"""Compare approximate and exact vector search.

An approximate index trades recall for latency. Both sides of that trade are
measured here rather than assumed: an index that saves two milliseconds while
losing five percent of correct neighbours is a poor exchange at small corpus
sizes, and the point at which it becomes worthwhile depends on scale.
"""

import statistics
import time

from pgvector.psycopg import register_vector

from cardterms.db import get_conn
from cardterms.embed.models import REGISTRY, encode_query

CHUNK_SET = "fixed_512_ov0"
MODEL = "bge-small"
TOP_K = 10
EF_VALUES = [10, 20, 40, 100, 200]

SEARCH_SQL = """
    SELECT c.id FROM embeddings_384 e
    JOIN chunks c ON c.id = e.chunk_id
    JOIN chunk_sets s ON s.id = c.chunk_set_id
    WHERE s.name = %s AND e.model = %s AND e.prefix_scheme = 'standard'
    ORDER BY e.vec <=> %s LIMIT %s
"""


def timed(conn, vector) -> tuple[list[int], float]:
    started = time.perf_counter()
    rows = conn.execute(SEARCH_SQL, (CHUNK_SET, MODEL, vector, TOP_K)).fetchall()
    return [r["id"] for r in rows], (time.perf_counter() - started) * 1000


def main() -> None:
    with get_conn() as conn:
        register_vector(conn)
        questions = [
            r["question"]
            for r in conn.execute("SELECT question FROM eval_questions").fetchall()
        ]
        vectors = [encode_query(REGISTRY[MODEL], q) for q in questions]

        # Ground truth: sequential scan, guaranteed correct neighbours.
        conn.execute("SET enable_indexscan = off")
        exact, exact_times = [], []
        for vector in vectors:
            ids, ms = timed(conn, vector)
            exact.append(set(ids))
            exact_times.append(ms)

        conn.execute("SET enable_indexscan = on")

        print(f"{len(vectors)} queries, top-{TOP_K}, 5,274 vectors\n")
        print(f"{'method':<16}{'recall':>9}{'median ms':>12}{'p95 ms':>10}")
        print("-" * 47)
        print(
            f"{'exact':<16}{1.000:>9.3f}"
            f"{statistics.median(exact_times):>12.1f}"
            f"{sorted(exact_times)[int(len(exact_times) * 0.95)]:>10.1f}"
        )

        for ef in EF_VALUES:
            conn.execute(f"SET hnsw.ef_search = {ef}")
            recalls, times = [], []
            for vector, truth in zip(vectors, exact, strict=True):
                ids, ms = timed(conn, vector)
                recalls.append(len(set(ids) & truth) / len(truth))
                times.append(ms)
            print(
                f"{'hnsw ef=' + str(ef):<16}{statistics.mean(recalls):>9.3f}"
                f"{statistics.median(times):>12.1f}"
                f"{sorted(times)[int(len(times) * 0.95)]:>10.1f}"
            )


if __name__ == "__main__":
    main()
