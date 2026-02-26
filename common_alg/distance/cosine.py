"""Cosine distance or cosine similarity."""

import math

import numpy as np


def cosine(d1: np.ndarray, d2: np.ndarray) -> float:
    """Calculate the Cosine distance between two points.

    Args:
        d1: First point as a numpy array.
        d2: Second point as a numpy array.

    Returns:
        The Cosine distance as a float.
    """
    dot_product = np.dot(d1, d2)
    norm_d1 = np.linalg.norm(d1)
    norm_d2 = np.linalg.norm(d2)
    cosine_similarity = dot_product / (norm_d1 * norm_d2)
    cosine_distance = 1 - cosine_similarity
    return float(cosine_distance)


# alternative implementation
class Cosine:
    """Alternative class-based implementation for computing cosine distance between vectors."""
    def __init__(self) -> None:
        """Initialize Cosine distance calculator."""
        pass

    def dot_product(self, d1: np.ndarray, d2: np.ndarray) -> float:
        """Compute dot product of two vectors."""
        if len(d1) != len(d2):
            msg = "Vectors must have the same dimension"
            raise ValueError(msg)
        dot_product: float = sum(a * b for a, b in zip(d1, d2, strict=True))
        return dot_product

    def magnitude(self, d: np.ndarray) -> float:
        """Compute L2 norm (magnitude) of a vector."""
        magnitude: float = math.sqrt(sum(x * x for x in d))
        return magnitude

    def cosine_similarity(self, d1: np.ndarray, d2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return self.dot_product(d1, d2) / (self.magnitude(d1) * self.magnitude(d2))

    def cosine_distance(self, d1: np.ndarray, d2: np.ndarray) -> float:
        """Compute cosine distance (1 - cosine_similarity) between two vectors."""
        cosine_similarity: float = self.cosine_similarity(d1, d2)
        return 1 - cosine_similarity

    def __call__(self, d1: np.ndarray, d2: np.ndarray) -> float:
        """Compute cosine distance between two vectors (callable interface)."""
        return self.cosine_distance(d1, d2)
