"""Leave-one-out splits and padded sequence datasets.

For a user sequence [i1, ..., in]:
  train: [i1, ..., i(n-2)]
  valid: history [i1, ..., i(n-2)] -> target i(n-1)
  test:  history [i1, ..., i(n-1)] -> target in
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def leave_one_out(sequences: list[list[int]]):
    """Return (train_seqs, valid, test); valid/test are (histories, targets)."""
    train = [s[:-2] for s in sequences]
    valid = ([s[:-2] for s in sequences], [s[-2] for s in sequences])
    test = ([s[:-1] for s in sequences], [s[-1] for s in sequences])
    return train, valid, test


def left_pad(seq: list[int], max_len: int) -> list[int]:
    """Keep the most recent max_len items and left-pad with 0."""
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


class NextItemDataset(Dataset):
    """Shifted input/target pairs for next-item training (SASRec-style).

    Input  [i1, ..., i(m-1)] predicts targets [i2, ..., im] at every position.
    """

    def __init__(self, train_seqs: list[list[int]], max_len: int):
        self.seqs = [s for s in train_seqs if len(s) >= 2]
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx][-(self.max_len + 1) :]
        inputs = left_pad(s[:-1], self.max_len)
        targets = left_pad(s[1:], self.max_len)
        return torch.tensor(inputs), torch.tensor(targets)


class NextItemPairs(Dataset):
    """(history, next item) pairs at every position of every sequence (TIGER-style).

    For [i1, ..., im]: [i1] -> i2, [i1, i2] -> i3, ..., [i1, ..., i(m-1)] -> im,
    keeping the most recent max_len history items.
    """

    def __init__(self, train_seqs: list[list[int]], max_len: int):
        self.seqs = train_seqs
        self.pairs = [(s, t) for s, seq in enumerate(train_seqs) for t in range(1, len(seq))]
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s, t = self.pairs[idx]
        seq = self.seqs[s]
        return torch.tensor(left_pad(seq[:t], self.max_len)), torch.tensor(seq[t])

    def history_lengths(self) -> np.ndarray:
        return np.array([min(t, self.max_len) for _, t in self.pairs])


class BucketBatchSampler(Sampler):
    """Batches of similar-length examples, in random order, so padding can be trimmed per batch.

    Each epoch shuffles, sorts by length within chunks of `bucket` batches, cuts
    those into batches, and shuffles the batches.
    """

    def __init__(self, lengths: np.ndarray, batch_size: int, bucket: int = 50):
        self.lengths, self.batch_size, self.bucket = lengths, batch_size, bucket

    def __len__(self):
        return -(-len(self.lengths) // self.batch_size)

    def __iter__(self):
        idx = np.random.permutation(len(self.lengths))
        chunk = self.batch_size * self.bucket
        batches = []
        for start in range(0, len(idx), chunk):
            part = idx[start : start + chunk]
            part = part[np.argsort(self.lengths[part], kind="stable")]
            batches += [part[i : i + self.batch_size].tolist() for i in range(0, len(part), self.batch_size)]
        for b in np.random.permutation(len(batches)):
            yield batches[b]


def pad_histories(histories: list[list[int]], max_len: int) -> torch.Tensor:
    return torch.tensor([left_pad(h, max_len) for h in histories])
