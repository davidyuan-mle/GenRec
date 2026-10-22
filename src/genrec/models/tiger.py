"""TIGER-style generative retrieval (Rajput et al., 2023).

Each item is a tuple of Semantic ID codes (e.g. 3 RQ-VAE codes + collision
suffix). A user's history is flattened into code tokens and read by a
Transformer encoder; the decoder generates the next item's codes one level at
a time. Code c at level l is token 1 + l*K + c, so every level has its own
token range; 0 is padding.

At inference, beam search is constrained by a trie of valid IDs, so every
generated tuple maps to a real item. Unlike TIGER, there is no user-ID token.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.data.sequences import pad_histories


class SemanticIDTrie(nn.Module):
    """Valid next codes for every valid ID prefix, and the item behind every full ID.

    A prefix (c_0, ..., c_{l-1}) is stored as the integer sum c_j * K^(l-1-j),
    which lets a batch of prefixes be looked up at once with searchsorted.
    """

    def __init__(self, codes: np.ndarray, k: int):
        super().__init__()
        self.k, self.n_levels = k, codes.shape[1]
        for level in range(self.n_levels):
            keys = self._keys_np(codes[:, :level])
            uniq, row = np.unique(keys, return_inverse=True)
            allowed = np.zeros((len(uniq), k), dtype=bool)
            allowed[row, codes[:, level]] = True
            self.register_buffer(f"keys_{level}", torch.from_numpy(uniq))
            self.register_buffer(f"allowed_{level}", torch.from_numpy(allowed))
        full = self._keys_np(codes)
        order = np.argsort(full)
        self.register_buffer("item_keys", torch.from_numpy(full[order]))
        self.register_buffer("item_ids", torch.from_numpy(order + 1))  # codes row i is item i + 1

    def _keys_np(self, prefix: np.ndarray) -> np.ndarray:
        keys = np.zeros(len(prefix), dtype=np.int64)
        for j in range(prefix.shape[1]):
            keys = keys * self.k + prefix[:, j]
        return keys

    def keys(self, prefix: torch.Tensor) -> torch.Tensor:
        keys = torch.zeros(prefix.shape[:-1], dtype=torch.long, device=prefix.device)
        for j in range(prefix.shape[-1]):
            keys = keys * self.k + prefix[..., j]
        return keys

    def allowed(self, prefix: torch.Tensor) -> torch.Tensor:
        """prefix: (..., l) codes -> (..., K) mask of valid next codes."""
        level = prefix.shape[-1]
        table, allowed = getattr(self, f"keys_{level}"), getattr(self, f"allowed_{level}")
        keys = self.keys(prefix)
        row = torch.searchsorted(table, keys).clamp(max=len(table) - 1)
        return allowed[row] & (table[row] == keys).unsqueeze(-1)

    def items(self, codes: torch.Tensor) -> torch.Tensor:
        """(..., L) full codes -> item indices, 0 where the tuple is not a valid ID."""
        keys = self.keys(codes)
        pos = torch.searchsorted(self.item_keys, keys).clamp(max=len(self.item_keys) - 1)
        return torch.where(self.item_keys[pos] == keys, self.item_ids[pos], 0)


class TIGER(nn.Module):
    def __init__(
        self,
        semantic_ids: np.ndarray,
        max_items: int = 20,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
        codebook_size: int = 256,
    ):
        """semantic_ids: (n_items + 1, L) codes from scripts/train_rqvae.py; row 0 is padding."""
        super().__init__()
        codes = semantic_ids[1:]
        assert codes.max() < codebook_size, "collision suffix must fit in the codebook"
        self.max_items, self.k, self.n_levels = max_items, codebook_size, codes.shape[1]
        offsets = 1 + codebook_size * np.arange(self.n_levels)
        item_tokens = np.vstack([np.zeros((1, self.n_levels), dtype=np.int64), codes + offsets])
        self.register_buffer("item_tokens", torch.from_numpy(item_tokens))
        self.register_buffer("offsets", torch.from_numpy(offsets))
        self.bos = 1 + codebook_size * self.n_levels
        self.trie = SemanticIDTrie(codes, codebook_size)

        self.tok_emb = nn.Embedding(self.bos + 1, d_model, padding_idx=0)
        self.src_pos = nn.Embedding(max_items * self.n_levels, d_model)
        self.tgt_pos = nn.Embedding(self.n_levels, d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout, batch_first=True, norm_first=True)
        encoder = nn.TransformerEncoder(layer, n_layers, nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.transformer = nn.Transformer(
            d_model, n_heads, n_layers, n_layers, dim_ff, dropout,
            custom_encoder=encoder, batch_first=True, norm_first=True,
        )
        self.heads = nn.ModuleList(nn.Linear(d_model, codebook_size) for _ in range(self.n_levels))

    def encode(self, histories: torch.Tensor):
        """histories: (B, max_items) left-padded item indices -> (memory, padding mask)."""
        # Drop leading columns that are padding for the whole batch; most histories are short.
        # Positions count from the right, so the most recent item always has the same one.
        first = int((histories != 0).any(0).int().argmax())
        src = self.item_tokens[histories[:, first:]].flatten(1)
        pad = src == 0
        n_pos = self.src_pos.num_embeddings
        positions = torch.arange(n_pos - src.size(1), n_pos, device=src.device)
        x = self.dropout(self.tok_emb(src) + self.src_pos(positions))
        return self.transformer.encoder(x, src_key_padding_mask=pad), pad

    def decode(self, memory: torch.Tensor, pad: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, l) codes generated so far -> (B, l + 1, d_model) decoder states."""
        bos = torch.full((codes.size(0), 1), self.bos, device=codes.device)
        tgt = torch.cat([bos, codes + self.offsets[: codes.size(1)]], dim=1)
        positions = torch.arange(tgt.size(1), device=tgt.device)
        y = self.dropout(self.tok_emb(tgt) + self.tgt_pos(positions))
        causal = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        return self.transformer.decoder(y, memory, tgt_mask=causal, tgt_is_causal=True, memory_key_padding_mask=pad)

    def loss(self, histories: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy of each target code given the history and the codes before it."""
        memory, pad = self.encode(histories)
        codes = self.item_tokens[targets] - self.offsets
        h = self.decode(memory, pad, codes[:, :-1])
        return sum(F.cross_entropy(head(h[:, l]), codes[:, l]) for l, head in enumerate(self.heads)) / self.n_levels

    @torch.no_grad()
    def generate(self, histories: torch.Tensor, beam: int) -> torch.Tensor:
        """Trie-constrained beam search -> (B, beam) item indices, best first (0 = no valid item)."""
        memory, pad = self.encode(histories)
        B = histories.size(0)
        codes = torch.zeros(B, 1, 0, dtype=torch.long, device=histories.device)
        scores = torch.zeros(B, 1, device=histories.device)
        for level, head in enumerate(self.heads):
            n_beams = codes.size(1)
            h = self.decode(
                memory.repeat_interleave(n_beams, 0), pad.repeat_interleave(n_beams, 0), codes.flatten(0, 1)
            )[:, -1]
            logp = F.log_softmax(head(h), dim=-1).view(B, n_beams, self.k)
            logp = logp.masked_fill(~self.trie.allowed(codes), -torch.inf)
            top = (scores.unsqueeze(-1) + logp).flatten(1).topk(min(beam, n_beams * self.k))
            parent = top.indices // self.k
            codes = torch.cat(
                [codes.gather(1, parent.unsqueeze(-1).expand(-1, -1, level)), (top.indices % self.k).unsqueeze(-1)],
                dim=-1,
            )
            scores = top.values
        items = self.trie.items(codes)
        return torch.where(torch.isfinite(scores), items, 0)

    @torch.no_grad()
    def recommend(self, histories: list[list[int]], k: int, beam: int = 20, batch_size: int = 256) -> list[list[int]]:
        self.eval()
        device = self.item_tokens.device
        # Batch users with similar history lengths so encode() can trim more padding.
        order = sorted(range(len(histories)), key=lambda i: len(histories[i]))
        out = [None] * len(histories)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            seqs = pad_histories([histories[i] for i in idx], self.max_items).to(device)
            for i, recs in zip(idx, self.generate(seqs, max(k, beam))[:, :k].tolist()):
                out[i] = recs
        return out
