import math

import pytest

from genrec.metrics import evaluate, mrr, ndcg_at_k, recall_at_k


def test_perfect_ranking():
    ranked = [[1, 2, 3], [4, 5, 6]]
    targets = [1, 4]
    assert recall_at_k(ranked, targets, 1) == 1.0
    assert ndcg_at_k(ranked, targets, 1) == 1.0
    assert mrr(ranked, targets) == 1.0


def test_miss_scores_zero():
    ranked = [[1, 2, 3]]
    targets = [9]
    assert recall_at_k(ranked, targets, 3) == 0.0
    assert ndcg_at_k(ranked, targets, 3) == 0.0
    assert mrr(ranked, targets) == 0.0


def test_second_position():
    ranked = [[7, 8, 9]]
    targets = [8]
    assert recall_at_k(ranked, targets, 1) == 0.0
    assert recall_at_k(ranked, targets, 2) == 1.0
    assert ndcg_at_k(ranked, targets, 2) == pytest.approx(1 / math.log2(3))
    assert mrr(ranked, targets) == 0.5


def test_evaluate_keys():
    out = evaluate([[1, 2]], [2], ks=(1, 2))
    assert set(out) == {"recall@1", "ndcg@1", "recall@2", "ndcg@2", "mrr"}
