# Minkowski distance
import numpy as np

def minkowski(d1: np.ndarray, d2: np.ndarray, p: int) -> float:
    '''Calculate the Minkowski distance between two points.
    args:
        d1: First point as a numpy array.
        d2: Second point as a numpy array.
        p: The order of the norm.
    returns:
        The Minkowski distance as a float.
    '''
    return np.sum(np.abs(d1 - d2) ** p) ** (1 / p)