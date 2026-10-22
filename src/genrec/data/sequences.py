"""Leave-one-out splits and padded sequence datasets.

For a user sequence [i1, ..., in]:
  train: [i1, ..., i(n-2)]
  valid: history [i1, ..., i(n-2)] -> target i(n-1)
  test:  history [i1, ..., i(n-1)] -> target in
"""

import torch
from torch.utils.data import Dataset


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


def pad_histories(histories: list[list[int]], max_len: int) -> torch.Tensor:
    return torch.tensor([left_pad(h, max_len) for h in histories])
