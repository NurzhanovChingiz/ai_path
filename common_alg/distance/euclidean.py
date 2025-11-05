# Euclidean distance

import numpy as np

def euclidean(d1: np.ndarray, d2: np.ndarray) -> float:
        return np.sqrt(np.sum((d1-d2)**2))
    
