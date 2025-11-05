import numpy as np
from euclidean import euclidean
from manhattan import manhattan
from minkowski import minkowski
from chebyshev import chebyshev
from cosine import cosine

if __name__ == "__main__":
    d1 = np.array([1, 2, 3])
    d2 = np.array([4, 5, 6])
    
    print('Euclidean distance:',euclidean(d1, d2))
    print('Manhattan distance:',manhattan(d1, d2))
    print('Minkowski distance:',minkowski(d1, d2, 2))
    print('Chebyshev distance:',chebyshev(d1, d2))
    print('Cosine distance:',cosine(d1, d2))