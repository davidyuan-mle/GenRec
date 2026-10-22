"""Encode item text into data/processed/<name>/embeddings/<model>.npy."""

import argparse

import numpy as np

from genrec.data.amazon import load_processed
from genrec.embeddings.text import DEFAULT_MODEL, embeddings_path, item_embeddings

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data/processed/beauty")
parser.add_argument("--model", default=DEFAULT_MODEL)
args = parser.parse_args()

_, items = load_processed(args.data_dir)
emb = item_embeddings(items, args.model)

out = embeddings_path(args.data_dir, args.model)
out.parent.mkdir(parents=True, exist_ok=True)
np.save(out, emb)
print(f"Wrote {out} with shape {emb.shape}")
