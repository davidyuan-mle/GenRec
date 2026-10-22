"""Semantic IDs from k-means: item embedding -> tuple of discrete codes.

Two ways to get L levels of codes:
  residual:     k-means on the embeddings, then on what each centroid leaves
                unexplained, and so on (RQ-KMeans, as in OneRec). Every level has
                one shared codebook, like RQ-VAE without the learned encoder.
  hierarchical: k-means, then k-means again inside each cluster (a tree).
                A code only means something under its parent prefix.

Both return (codes, recon): codes is (n, L) ints, and recon is the quantized
embedding each item's code tuple stands for.
"""

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans


def residual_kmeans(emb: np.ndarray, k: int, n_levels: int = 3, seed: int = 42):
    codes, recon = [], np.zeros_like(emb)
    for level in range(n_levels):
        km = KMeans(k, random_state=seed + level).fit(emb - recon)
        codes.append(km.labels_)
        recon += km.cluster_centers_[km.labels_]
    return np.stack(codes, axis=1), recon


def hierarchical_kmeans(emb: np.ndarray, k: int, n_levels: int = 3, seed: int = 42):
    codes, recon = np.zeros((len(emb), n_levels), dtype=int), np.zeros_like(emb)

    def split(idx: np.ndarray, level: int):
        if len(idx) <= k:
            # One cluster per item; k-means with k >= n points would do the same.
            labels, centers = np.arange(len(idx)), emb[idx]
        else:
            km = KMeans(k, random_state=seed).fit(emb[idx])
            labels, centers = km.labels_, km.cluster_centers_
        codes[idx, level] = labels
        if level == n_levels - 1:
            recon[idx] = centers[labels]
            return
        for c in np.unique(labels):
            split(idx[labels == c], level + 1)

    split(np.arange(len(emb)), 0)
    return codes, recon


def add_collision_suffix(codes: np.ndarray) -> np.ndarray:
    """Append a code numbering items that share a code tuple (0, 1, ...), as in TIGER."""
    seen = Counter()
    suffix = np.zeros(len(codes), dtype=int)
    for i, key in enumerate(map(tuple, codes)):
        suffix[i] = seen[key]
        seen[key] += 1
    return np.column_stack([codes, suffix])


def code_stats(codes: np.ndarray, emb: np.ndarray | None = None, recon: np.ndarray | None = None) -> dict:
    """Collision and codebook-usage stats for (n, L) codes, before any collision suffix."""
    group_sizes = np.array(list(Counter(map(tuple, codes)).values()))
    out = {
        "unique_ids": len(group_sizes),
        "collision_rate": group_sizes[group_sizes > 1].sum() / len(codes),  # items sharing their ID
        "max_group": int(group_sizes.max()),
    }
    for level in range(codes.shape[1]):
        p = np.bincount(codes[:, level]) / len(codes)
        p = p[p > 0]
        out[f"l{level + 1}_used"] = len(p)
        out[f"l{level + 1}_perplexity"] = float(np.exp(-(p * np.log(p)).sum()))
    if emb is not None and recon is not None:
        cos = (emb * recon).sum(1) / (np.linalg.norm(recon, axis=1) + 1e-12)
        out["recon_cosine"] = float(cos.mean())
    return out


def quantized_embeddings(recon: np.ndarray) -> np.ndarray:
    """L2-normalize recon and prepend a zero padding row, matching item_embeddings()."""
    q = recon / (np.linalg.norm(recon, axis=1, keepdims=True) + 1e-12)
    return np.vstack([np.zeros((1, q.shape[1]), dtype=np.float32), q.astype(np.float32)])
