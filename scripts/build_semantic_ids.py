"""Build k-means Semantic IDs from item embeddings.

Writes data/processed/<name>/semantic_ids/<method>_kmeans-<levels>x<k>.npy, with
codes of shape (n_items + 1, levels + 1): the last column is the collision suffix,
and row 0 (padding) is -1. Also writes <same name>_quantized.npy, the quantized
item embeddings, for semantic kNN.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from genrec.embeddings.text import DEFAULT_MODEL, embeddings_path
from genrec.tokenizers.kmeans import (
    add_collision_suffix,
    code_stats,
    hierarchical_kmeans,
    quantized_embeddings,
    residual_kmeans,
)

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data/processed/beauty")
parser.add_argument("--model", default=DEFAULT_MODEL, help="embedding model from embed_items.py")
parser.add_argument("--method", choices=["residual", "hierarchical"], default="residual")
parser.add_argument("--k", type=int, default=256, help="codebook size per level")
parser.add_argument("--levels", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

emb = np.load(embeddings_path(args.data_dir, args.model))[1:]  # drop padding row
build = residual_kmeans if args.method == "residual" else hierarchical_kmeans
codes, recon = build(emb, args.k, args.levels, args.seed)

ids = np.vstack([np.full((1, args.levels + 1), -1), add_collision_suffix(codes)])
out_dir = Path(args.data_dir) / "semantic_ids"
out_dir.mkdir(parents=True, exist_ok=True)
name = f"{args.method}_kmeans-{args.levels}x{args.k}"
np.save(out_dir / f"{name}.npy", ids)
np.save(out_dir / f"{name}_quantized.npy", quantized_embeddings(recon))

print(json.dumps(code_stats(codes, emb, recon), indent=2))
print(f"Wrote {out_dir / name}.npy with shape {ids.shape}")
