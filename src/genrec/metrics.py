"""Ranking metrics for leave-one-out evaluation (one held-out target per user).

`ranked` is a list of top-K item lists (best first), one per user.
`targets` is the held-out item for each user.
"""

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[Sequence], targets: Sequence, k: int) -> float:
    hits = sum(t in r[:k] for r, t in zip(ranked, targets, strict=True))
    return hits / len(targets)


def ndcg_at_k(ranked: Sequence[Sequence], targets: Sequence, k: int) -> float:
    # With a single relevant item, IDCG = 1, so NDCG = 1 / log2(rank + 1).
    total = 0.0
    for r, t in zip(ranked, targets, strict=True):
        top = list(r[:k])
        if t in top:
            total += 1.0 / math.log2(top.index(t) + 2)
    return total / len(targets)


def mrr(ranked: Sequence[Sequence], targets: Sequence) -> float:
    total = 0.0
    for r, t in zip(ranked, targets, strict=True):
        r = list(r)
        if t in r:
            total += 1.0 / (r.index(t) + 1)
    return total / len(targets)


def evaluate(ranked: Sequence[Sequence], targets: Sequence, ks=(5, 10)) -> dict:
    out = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, targets, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, targets, k)
    out["mrr"] = mrr(ranked, targets)
    return out
