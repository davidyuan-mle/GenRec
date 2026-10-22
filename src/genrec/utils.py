"""Shared helpers: seeding, device selection, config loading."""

import random
from pathlib import Path

import numpy as np
import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def get_device() -> str:
    """Prefer Apple MPS, then CUDA, then CPU."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
