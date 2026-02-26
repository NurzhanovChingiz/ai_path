"""Jaccard distance or Jaccard similarity."""

import numpy as np


def jaccard(d1: np.ndarray, d2: np.ndarray) -> float:
    """Calculate the Jaccard distance between two binary vectors.

    Args:
        d1: First point as a numpy array (binary vector).
        d2: Second point as a numpy array (binary vector).
    returns:
        The Jaccard distance as a float.
    """
    intersection = np.sum(np.minimum(d1, d2))
    union = np.sum(np.maximum(d1, d2))
    jaccard_similarity = intersection / union if union != 0 else 0.0
    jaccard_distance = 1 - jaccard_similarity
    return jaccard_distance
