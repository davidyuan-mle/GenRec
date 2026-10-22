"""Non-learned baselines."""

from collections import Counter


class Popularity:
    """Recommend the most frequent training items to every user."""

    def fit(self, train_seqs: list[list[int]]) -> "Popularity":
        counts = Counter(i for s in train_seqs for i in s)
        self.ranking = [i for i, _ in counts.most_common()]
        return self

    def recommend(self, histories: list[list[int]], k: int) -> list[list[int]]:
        top = self.ranking[:k]
        return [top for _ in histories]
