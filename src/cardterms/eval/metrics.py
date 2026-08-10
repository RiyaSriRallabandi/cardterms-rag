"""Retrieval metrics.

Hit rate and recall answer different questions and both are reported. For a
question with a single answer, the system succeeds if any relevant passage
reaches the context, which is hit rate. For a comparison question, every side
must be retrieved, which is recall. Reporting only one would flatter or
penalise whole categories unfairly.
"""

import math


def hit_rate_at_k(retrieved: list[int], relevant: dict[int, int], k: int) -> float:
    """1.0 if any relevant chunk is in the top k."""
    if not relevant:
        return 0.0
    return float(any(chunk_id in relevant for chunk_id in retrieved[:k]))


def recall_at_k(retrieved: list[int], relevant: dict[int, int], k: int) -> float:
    """Proportion of relevant chunks appearing in the top k."""
    if not relevant:
        return 0.0
    found = sum(1 for chunk_id in retrieved[:k] if chunk_id in relevant)
    return found / len(relevant)


def precision_at_k(retrieved: list[int], relevant: dict[int, int], k: int) -> float:
    """Proportion of the top k that is relevant.

    Reported for completeness but not used to choose between systems: with two
    relevant chunks, precision@10 cannot exceed 0.2 regardless of ranking
    quality, so it penalises a k chosen for other reasons.
    """
    if k == 0:
        return 0.0
    return sum(1 for chunk_id in retrieved[:k] if chunk_id in relevant) / k


def reciprocal_rank(retrieved: list[int], relevant: dict[int, int]) -> float:
    """1 / rank of the first relevant chunk, else 0.

    Rank matters beyond mere presence: language models attend most reliably to
    the start and end of a context window.
    """
    for index, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: dict[int, int], k: int) -> float:
    """Rank-weighted gain, normalised against the best possible ordering."""
    if not relevant:
        return 0.0

    gain = sum(
        relevant.get(chunk_id, 0) / math.log2(index + 1)
        for index, chunk_id in enumerate(retrieved[:k], start=1)
    )
    ideal = sum(
        grade / math.log2(index + 1)
        for index, grade in enumerate(
            sorted(relevant.values(), reverse=True)[:k], start=1
        )
    )
    return gain / ideal if ideal else 0.0


def document_hit_rate(
    retrieved_docs: list[int], relevant_docs: set[int], k: int
) -> float:
    """1.0 if any retrieved chunk came from a document that answers.

    Separates entity confusion from chunking failure: a system that finds the
    right document but the wrong passage has a different problem from one that
    finds the wrong issuer entirely.
    """
    if not relevant_docs:
        return 0.0
    return float(any(doc_id in relevant_docs for doc_id in retrieved_docs[:k]))


def score_question(
    retrieved: list[int],
    retrieved_docs: list[int],
    relevant: dict[int, int],
    relevant_docs: set[int],
    k_values: list[int],
) -> dict[str, float]:
    scores: dict[str, float] = {"mrr": reciprocal_rank(retrieved, relevant)}
    for k in k_values:
        scores[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant, k)
        scores[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        scores[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
        scores[f"doc_hit_rate@{k}"] = document_hit_rate(
            retrieved_docs, relevant_docs, k
        )
    return scores
