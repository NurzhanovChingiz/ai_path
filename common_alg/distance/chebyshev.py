# Chebyshev distance or L∞ norm
import numpy as np

def chebyshev(d1: np.ndarray, d2: np.ndarray) -> float:
    '''Calculate the Chebyshev distance or L∞ norm between two points.
    args:
        d1: First point as a numpy array.
        d2: Second point as a numpy array.
    returns:
        The Chebyshev distance as a float.
    '''
    return float(np.max(np.abs(d1 - d2)))