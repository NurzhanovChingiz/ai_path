"""Insertion sort implementation."""

from collections.abc import Generator


def insertion_sort(a: list[int]) -> Generator[list[int], None, None]:
    """Sort a list in-place using insertion sort, yielding the array after each step."""
    arr = a
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
            yield arr
        arr[j + 1] = key
        yield arr
