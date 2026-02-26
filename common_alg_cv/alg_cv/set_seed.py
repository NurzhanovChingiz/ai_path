"""Seed setting for reproducibility."""

import random

import numpy as np
import torch


def set_seed(SEED: int = 42) -> None:
    """Set random seeds for PyTorch, NumPy, and Python for reproducibility."""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print(f'Random Seed : {SEED}')
