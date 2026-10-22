"""Non-learned baselines."""

from collections import Counter

import numpy as np


class Popularity:
    """Recommend the most frequent training items to every user."""

    def fit(self, train_seqs: list[list[int]]) -> "Popularity":
        counts = Counter(i for s in train_seqs for i in s)
        self.ranking = [i for i, _ in counts.most_common()]
        return self

    def recommend(self, histories: list[list[int]], k: int) -> list[list[int]]:
        top = self.ranking[:k]
        return [top for _ in histories]


class SemanticKNN:
    """Recommend items whose text embedding is closest to the user's recent items.

    The user profile is a recency-weighted sum of the last `n_recent` item
    embeddings (weight decay**age, age 0 = most recent). Items are ranked by
    cosine similarity to it. History items are excluded, since otherwise the
    last item, with similarity ~1, would always come first.
    """

    def __init__(self, item_emb: np.ndarray, n_recent: int = 5, decay: float = 0.5):
        self.item_emb = item_emb  # (n_items + 1, dim), L2-normalized, row 0 = padding
        self.n_recent = n_recent
        self.decay = decay

    def recommend(self, histories: list[list[int]], k: int, batch_size: int = 1024) -> list[list[int]]:
        out = []
        for start in range(0, len(histories), batch_size):
            batch = histories[start : start + batch_size]
            profiles = np.stack([self._profile(h) for h in batch])
            scores = profiles @ self.item_emb.T
            scores[:, 0] = -np.inf  # never recommend padding
            for row, h in enumerate(batch):
                scores[row, h] = -np.inf
            top = np.argpartition(-scores, k, axis=1)[:, :k]
            order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
            out.extend(np.take_along_axis(top, order, axis=1).tolist())
        return out

    def _profile(self, history: list[int]) -> np.ndarray:
        recent = history[-self.n_recent :]
        weights = self.decay ** np.arange(len(recent))[::-1]
        return weights @ self.item_emb[recent]
