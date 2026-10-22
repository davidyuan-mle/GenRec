"""Train an RQ-VAE tokenizer and write its Semantic IDs.

Writes data/processed/<name>/semantic_ids/<cfg name>.npy (codes with collision
suffix, row 0 = -1), <cfg name>_quantized.npy (decoded embeddings for semantic
kNN) and <cfg name>_history.json (training curve), in the same format as
scripts/build_semantic_ids.py.

  python scripts/train_rqvae.py --config configs/rqvae_beauty.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np

from genrec.tokenizers.kmeans import add_collision_suffix, code_stats, quantized_embeddings
from genrec.tokenizers.rqvae import fit_rqvae
from genrec.utils import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--device", default="cpu", help="the model is small enough that CPU is as fast as MPS")
args = parser.parse_args()
cfg = load_config(args.config)

emb = np.load(cfg["embeddings"])[1:]  # drop padding row
codes, recon, history = fit_rqvae(emb, cfg, args.device)

ids = np.vstack([np.full((1, cfg["levels"] + 1), -1), add_collision_suffix(codes)])
out_dir = Path(cfg["data_dir"]) / "semantic_ids"
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / f"{cfg['name']}.npy", ids)
np.save(out_dir / f"{cfg['name']}_quantized.npy", quantized_embeddings(recon))
(out_dir / f"{cfg['name']}_history.json").write_text(json.dumps(history))

print(json.dumps(code_stats(codes, emb, recon), indent=2))
print(f"Wrote {out_dir / cfg['name']}.npy with shape {ids.shape}")
