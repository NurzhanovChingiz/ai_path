"""Quick sort implementation."""

from collections.abc import Generator


def quick_sort(a: list[int]) -> Generator[list[int], None, None]:
    """Sort a list in-place using quick sort, yielding the array after each partition."""
    arr = a

    def _q(left: int, r: int) -> Generator[list[int], None, None]:
        if left >= r:
            return
        pivot = arr[r]
        i = left
        for j in range(left, r):
            if arr[j] < pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
            yield arr
        arr[i], arr[r] = arr[r], arr[i]
        yield arr
        yield from _q(left, i - 1)
        yield from _q(i + 1, r)

    yield from _q(0, len(arr) - 1)
