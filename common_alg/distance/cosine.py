# Cosine distance or cosine similarity
import numpy as np

def cosine(d1: np.ndarray, d2: np.ndarray) -> float:
    '''Calculate the Cosine distance between two points.
    args:
        d1: First point as a numpy array.
        d2: Second point as a numpy array.
    returns:
        The Cosine distance as a float.
    '''
    dot_product = np.dot(d1, d2)
    norm_d1 = np.linalg.norm(d1)
    norm_d2 = np.linalg.norm(d2)
    cosine_similarity = dot_product / (norm_d1 * norm_d2)
    cosine_distance = 1 - cosine_similarity
    return cosine_distance