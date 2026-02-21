# Euclidean distance or L2 norm

import numpy as np


def euclidean(d1: np.ndarray, d2: np.ndarray) -> float:
    '''Calculate the Euclidean distance or L2 norm between two points.
    args:
        d1: First point as a numpy array.
        d2: Second point as a numpy array.
    returns:
        The Euclidean distance as a float.
    '''
    return float(np.sqrt(np.sum((d1 - d2)**2)))
