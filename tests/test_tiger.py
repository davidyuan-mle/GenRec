import numpy as np
import torch

from genrec.data.sequences import NextItemPairs
from genrec.models.tiger import TIGER, SemanticIDTrie

# Items 1-4 as 3-code IDs; rows are item indices, row 0 is padding.
SEMANTIC_IDS = np.array([[-1, -1, -1], [0, 1, 0], [0, 1, 1], [0, 2, 0], [3, 0, 0]])


def test_next_item_pairs():
    ds = NextItemPairs([[1, 2, 3], [7]], max_len=2)
    assert len(ds) == 2
    assert [ds[i][0].tolist() for i in range(2)] == [[0, 1], [1, 2]]
    assert [ds[i][1].item() for i in range(2)] == [2, 3]


def test_trie_allowed_and_items():
    trie = SemanticIDTrie(SEMANTIC_IDS[1:], k=4)
    assert trie.allowed(torch.zeros(1, 0, dtype=torch.long)).nonzero()[:, 1].tolist() == [0, 3]
    assert trie.allowed(torch.tensor([[0, 1]])).nonzero()[:, 1].tolist() == [0, 1]
    assert not trie.allowed(torch.tensor([[1, 1]])).any()  # invalid prefix
    assert trie.items(torch.tensor([[0, 2, 0], [3, 0, 0], [3, 0, 1]])).tolist() == [3, 4, 0]


def test_tiger_generates_only_valid_items():
    torch.manual_seed(0)
    model = TIGER(SEMANTIC_IDS, max_items=3, d_model=16, n_heads=2, n_layers=1, dim_ff=32, codebook_size=4)
    loss = model.loss(torch.tensor([[0, 1, 2], [0, 0, 3]]), torch.tensor([3, 4]))
    assert torch.isfinite(loss)
    recs = model.recommend([[1, 2], [3]], k=4, beam=4)
    assert all(sorted(r) == [1, 2, 3, 4] for r in recs)  # 4 valid items, all reachable with beam 4


def test_bucket_batch_sampler_covers_all_and_groups_lengths():
    from genrec.data.sequences import BucketBatchSampler

    lengths = np.array([1, 5, 1, 5, 1, 5, 1, 5])
    batches = list(BucketBatchSampler(lengths, batch_size=4, bucket=2))
    assert sorted(i for b in batches for i in b) == list(range(8))
    assert all(len(set(lengths[b])) == 1 for b in batches)
