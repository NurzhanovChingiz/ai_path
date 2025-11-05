# Manhattan distance
import numpy as np

def manhattan(d1: np.ndarray, d2: np.ndarray) -> float:
    return np.sum(np.abs(d1 - d2))