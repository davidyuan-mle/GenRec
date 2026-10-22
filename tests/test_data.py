import numpy as np
import pandas as pd
import torch

from genrec.data.amazon import kcore_filter
from genrec.data.sequences import NextItemDataset, leave_one_out, left_pad
from genrec.models.baselines import SemanticKNN
from genrec.models.sasrec import SASRec


def test_kcore_filter_is_iterative():
    # u3 has 1 interaction; dropping it leaves item c with 1, which must also go.
    df = pd.DataFrame(
        {"user": ["u1", "u1", "u2", "u2", "u3"], "item": ["a", "b", "a", "b", "c"], "timestamp": range(5)}
    )
    out = kcore_filter(df, k=2)
    assert set(out["user"]) == {"u1", "u2"}
    assert set(out["item"]) == {"a", "b"}


def test_leave_one_out():
    train, valid, test = leave_one_out([[1, 2, 3, 4, 5]])
    assert train == [[1, 2, 3]]
    assert valid == ([[1, 2, 3]], [4])
    assert test == ([[1, 2, 3, 4]], [5])


def test_left_pad_truncates_to_recent():
    assert left_pad([1, 2], 4) == [0, 0, 1, 2]
    assert left_pad([1, 2, 3, 4, 5], 3) == [3, 4, 5]


def test_next_item_dataset_shift():
    ds = NextItemDataset([[1, 2, 3], [7]], max_len=4)
    assert len(ds) == 1  # sequences shorter than 2 are dropped
    inputs, targets = ds[0]
    assert inputs.tolist() == [0, 0, 1, 2]
    assert targets.tolist() == [0, 0, 2, 3]


def test_sasrec_ignores_left_padding():
    torch.manual_seed(0)
    model = SASRec(n_items=10, max_len=6, hidden=16, n_heads=2, dropout=0.0).eval()
    seqs = torch.tensor([[0, 0, 0, 3, 4, 5]])
    h = model.encode(seqs)
    assert not torch.isnan(h).any()
    # Perturbing what the padded positions carry must not change real positions.
    with torch.no_grad():
        model.pos_emb.weight[:3] += 10.0
    h2 = model.encode(seqs)
    assert torch.allclose(h[:, 3:], h2[:, 3:], atol=1e-5)


def test_semantic_knn_follows_recent_items_and_skips_history():
    # Items 1, 2 point along x; 3, 5 along y; 4 points away from y.
    emb = np.array([[0, 0], [1, 0], [1, 0], [0, 1], [0.6, -0.8], [0, 1]], dtype=np.float32)
    model = SemanticKNN(emb, n_recent=2, decay=0.5)
    # History items 1 and 3 score highest but are skipped; the most recent one decides the order.
    assert model.recommend([[3, 1]], k=2) == [[2, 5]]
    assert model.recommend([[1, 3]], k=2) == [[5, 2]]
