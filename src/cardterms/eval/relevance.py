"""Resolve labelled character spans to the chunks that contain them.

A chunk is relevant when it lies in the same document as a label and their
character ranges overlap. Computing relevance this way keeps the evaluation
set independent of chunking: one set of labels scores every chunk set.
"""

from cardterms.db import get_conn


def relevant_chunks(conn, chunk_set_name: str) -> dict[int, dict[int, int]]:
    """Return {question_id: {chunk_id: relevance_grade}}.

    Where a chunk overlaps several labels, the highest grade applies.
    """
    rows = conn.execute(
        """
        SELECT l.question_id, c.id AS chunk_id, l.relevance
        FROM eval_labels l
        JOIN chunks c ON c.doc_id = l.doc_id
        JOIN chunk_sets s ON s.id = c.chunk_set_id
        WHERE s.name = %s
          AND NOT c.is_parent
          AND c.char_start < l.char_end
          AND c.char_end   > l.char_start
        """,
        (chunk_set_name,),
    ).fetchall()

    resolved: dict[int, dict[int, int]] = {}
    for row in rows:
        grades = resolved.setdefault(row["question_id"], {})
        grades[row["chunk_id"]] = max(grades.get(row["chunk_id"], 0), row["relevance"])
    return resolved


def relevant_documents(conn) -> dict[int, set[int]]:
    """Return {question_id: {doc_id}} — which documents answer each question."""
    rows = conn.execute("SELECT question_id, doc_id FROM eval_labels").fetchall()
    resolved: dict[int, set[int]] = {}
    for row in rows:
        resolved.setdefault(row["question_id"], set()).add(row["doc_id"])
    return resolved


def coverage_report(chunk_set_name: str) -> list[dict]:
    """Questions whose labels resolve to no chunk in this chunk set."""
    with get_conn() as conn:
        resolved = relevant_chunks(conn, chunk_set_name)
        questions = conn.execute(
            """
            SELECT q.id, q.question_uid, q.category
            FROM eval_questions q
            WHERE EXISTS (SELECT 1 FROM eval_labels l WHERE l.question_id = q.id)
            """
        ).fetchall()
    return [dict(q) for q in questions if not resolved.get(q["id"])]
