from collections.abc import Callable

import numpy as np
import pytest

from common_alg.distance import (chebyshev, cosine, euclidean, jaccard,
                                 manhattan, minkowski)

ATOL: float = 1e-5  # 0.001%
RTOL: float = 1e-3  # 0.1%

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def inputs_ndarray() -> list[tuple[np.ndarray, np.ndarray]]:
    np.random.seed(42)
    d1_cases = [
        np.array([1, 2, 3]),
        np.array([4, 5, 6]) * 100,
    ]
    d2_cases = [
        np.array([4, 5, 6]),
        np.array([7, 8, 9]) * 100,
    ]
    inputs = []
    for d1 in d1_cases:
        for d2 in d2_cases:
            inputs.append((d1, d2))
    return inputs


def run_test(custom_cls: Callable,
             input_ndarray: tuple[np.ndarray,
                                  np.ndarray]) -> None:
    custom_cls(input_ndarray[0], input_ndarray[1])

# ── Test Cases ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("custom_cls", [
    euclidean.euclidean,
    cosine.cosine,
    cosine.Cosine().cosine_distance,
    minkowski.minkowski,
    manhattan.manhattan,
    jaccard.jaccard,
    chebyshev.chebyshev,
])
class TestDistance:
    def test_matches_pytorch(self,
                             custom_cls: Callable[[np.ndarray,
                                                   np.ndarray],
                                                  float],
                             inputs_ndarray: list[tuple[np.ndarray,
                                                        np.ndarray]]) -> None:
        for input_ndarray in inputs_ndarray:
            run_test(custom_cls, input_ndarray)
