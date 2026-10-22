"""SASRec: self-attentive sequential recommendation (Kang & McAuley, 2018).

ID-based baseline: a causal Transformer over item-ID embeddings. The last
hidden state is scored against all item embeddings (shared input/output
table). Trained with full softmax cross-entropy, which is cheap at this
catalog size and usually stronger than the paper's sampled BCE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.data.sequences import pad_histories


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        max_len: int = 50,
        hidden: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.max_len = max_len
        self.n_heads = n_heads
        self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            hidden, n_heads, dim_feedforward=hidden * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.register_buffer(
            "causal_mask", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        )
        # Small init: the item table is also the output layer, so N(0, 1) gives huge initial logits.
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def encode(self, seqs: torch.Tensor) -> torch.Tensor:
        """seqs: (B, L) left-padded item indices -> (B, L, H) hidden states."""
        positions = torch.arange(seqs.size(1), device=seqs.device)
        x = self.item_emb(seqs) + self.pos_emb(positions)
        x = self.dropout(x)
        # Block future and padded keys. Keep the diagonal open so padded rows
        # (which only see padding under left padding) don't become all-masked.
        L = seqs.size(1)
        mask = self.causal_mask[:L, :L] | (seqs == 0).unsqueeze(1)
        mask = mask & ~torch.eye(L, dtype=torch.bool, device=seqs.device)
        mask = mask.repeat_interleave(self.n_heads, dim=0)
        x = self.encoder(x, mask=mask)
        return self.norm(x)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.item_emb.weight.T

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.logits(self.encode(inputs))
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0)

    @torch.no_grad()
    def recommend(self, histories: list[list[int]], k: int, batch_size: int = 512) -> list[list[int]]:
        self.eval()
        device = self.item_emb.weight.device
        out = []
        for start in range(0, len(histories), batch_size):
            seqs = pad_histories(histories[start : start + batch_size], self.max_len).to(device)
            scores = self.logits(self.encode(seqs)[:, -1])
            scores[:, 0] = -torch.inf  # never recommend padding
            out.extend(scores.topk(k).indices.tolist())
        return out
