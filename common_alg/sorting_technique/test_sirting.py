from collections.abc import Callable

import numpy as np
import pytest

from common_alg.sorting_technique import (bubble_sort, insertion_sort,
                                          merge_sort, quick_sort)

ATOL: float = 1e-5  # 0.001%
RTOL: float = 1e-3  # 0.1%

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def inputs_list() -> list[list[int]]:
    np.random.seed(42)
    inputs = [
        [3, 2, 1],
        [1, 2, 3],
        [1, 3, 2],
    ]
    return inputs


def assert_close_all(
        custom_cls: Callable[[list[int]], list[int]], input_list: list[int]) -> None:
    '''
    Assert that the custom sorting function matches the numpy sorting function
    :param custom_cls: The custom sorting function
    :param input_list: The input list to be sorted
    '''
    expected = sorted(input_list)
    arr = list(input_list)
    for _ in custom_cls(arr):
        pass
    np.testing.assert_equal(arr, expected)


def run_sorting(
        custom_cls: Callable[[list[int]], list[int]], input_list: list[int]) -> None:
    '''
    Run the custom sorting function (exhausts the generator)
    :param custom_cls: The custom sorting function
    :param input_list: The input list to be sorted
    '''
    for _ in custom_cls(input_list):
        pass

# ── Test Cases ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("custom_cls", [
    bubble_sort.bubble_sort,
    insertion_sort.insertion_sort,
    merge_sort.merge_sort,
    quick_sort.quick_sort,
])
class TestSorting:
    def test_matches_numpy(self,
                           custom_cls: Callable[[list[int]],
                                                list[int]],
                           inputs_list: list[list[int]]) -> None:
        for input_list in inputs_list:
            assert_close_all(custom_cls, input_list)

    def run_test(self,
                 custom_cls: Callable[[list[int]],
                                      list[int]],
                 inputs_list: list[list[int]]) -> None:
        for input_list in inputs_list:
            run_sorting(custom_cls, input_list)
