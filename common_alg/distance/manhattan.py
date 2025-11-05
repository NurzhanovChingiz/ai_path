# Manhattan distance or L1 norm
import numpy as np

def manhattan(d1: np.ndarray, d2: np.ndarray) -> float:
    '''Calculate the Manhattan distance or L1 norm between two points.
    args:
        d1: First point as a numpy array.
        d2: Second point as a numpy array.
    returns:
        The Manhattan distance as a float.
    '''
    return np.sum(np.abs(d1 - d2))