"""Item text -> L2-normalized dense embeddings with a sentence-transformer.

TIGER encodes item text with Sentence-T5 XXL; sentence-t5-base is the
laptop-sized version of the same encoder.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from genrec.utils import get_device

DEFAULT_MODEL = "sentence-transformers/sentence-t5-base"


def encode_texts(texts: list[str], model_name: str = DEFAULT_MODEL, batch_size: int = 128) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=get_device())
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    return emb.astype(np.float32)


def item_embeddings(items: pd.DataFrame, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Return an (n_items + 1, dim) array where row i embeds item i; row 0 (padding) is zeros."""
    assert (items.index == np.arange(1, len(items) + 1)).all(), "items must be indexed 1..N in order"
    emb = encode_texts(items["text"].tolist(), model_name)
    return np.vstack([np.zeros((1, emb.shape[1]), dtype=np.float32), emb])


def embeddings_path(data_dir: str | Path, model_name: str = DEFAULT_MODEL) -> Path:
    return Path(data_dir) / "embeddings" / f"{model_name.split('/')[-1]}.npy"
