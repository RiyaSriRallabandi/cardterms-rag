"""Metrics verified against a hand-computed example.

retrieved: [10, 11, 12, 13, 14]
relevant:  {11: grade 2, 13: grade 1}
"""

import math

import pytest

from cardterms.eval.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = [10, 11, 12, 13, 14]
RELEVANT = {11: 2, 13: 1}


def test_hit_rate():
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 1) == 0.0  # 10 is not relevant
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 2) == 1.0  # 11 is
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 5) == 1.0


def test_recall():
    assert recall_at_k(RETRIEVED, RELEVANT, 1) == 0.0  # 0 of 2
    assert recall_at_k(RETRIEVED, RELEVANT, 2) == 0.5  # 1 of 2
    assert recall_at_k(RETRIEVED, RELEVANT, 5) == 1.0  # 2 of 2


def test_precision():
    assert precision_at_k(RETRIEVED, RELEVANT, 2) == 0.5  # 1 of 2 positions
    assert precision_at_k(RETRIEVED, RELEVANT, 4) == 0.5  # 2 of 4 positions


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, RELEVANT) == 0.5  # first hit at rank 2
    assert reciprocal_rank([10, 12, 14], RELEVANT) == 0.0


def test_ndcg():
    # DCG  = 2/log2(3) + 1/log2(5)
    # IDCG = 2/log2(2) + 1/log2(3)
    dcg = 2 / math.log2(3) + 1 / math.log2(5)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(dcg / idcg)


def test_empty_relevant_set():
    """Unanswerable questions have no relevant chunks and score zero."""
    assert hit_rate_at_k(RETRIEVED, {}, 5) == 0.0
    assert recall_at_k(RETRIEVED, {}, 5) == 0.0
    assert ndcg_at_k(RETRIEVED, {}, 5) == 0.0
