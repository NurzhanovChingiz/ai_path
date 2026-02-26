# Quick Sort

from typing import Generator, List


def quick_sort(a: List[int]) -> Generator[List[int], None, None]:
    arr = a

    def _q(l: int, r: int) -> Generator[List[int], None, None]:
        if l >= r:
            return
        pivot = arr[r]
        i = l
        for j in range(l, r):
            if arr[j] < pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
            yield arr
        arr[i], arr[r] = arr[r], arr[i]
        yield arr
        yield from _q(l, i - 1)
        yield from _q(i + 1, r)

    yield from _q(0, len(arr) - 1)
