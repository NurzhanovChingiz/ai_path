import numpy as np
from euclidean import euclidean


if __name__ == "__main__":
    d1 = np.array([1, 2, 3])
    d2 = np.array([4, 5, 6])
    print(euclidean(d1, d2))