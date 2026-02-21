# Merge Sort

from typing import List, Generator


def merge_sort(a: List[int]) -> Generator[List[int], None, None]:
    arr = a
    aux = [0] * len(arr)

    def _merge_sort(l: int, r: int) -> Generator[List[int], None, None]:
        if r - l <= 1:
            return
        m = (l + r) // 2
        yield from _merge_sort(l, m)
        yield from _merge_sort(m, r)
        i, j, k = l, m, l
        while i < m and j < r:
            if arr[i] < arr[j]:
                aux[k] = arr[i]
                i += 1
            else:
                aux[k] = arr[j]
                j += 1
            k += 1
            yield arr
        while i < m:
            aux[k] = arr[i]
            i += 1
            k += 1
            yield arr
        while j < r:
            aux[k] = arr[j]
            j += 1
            k += 1
            yield arr
        for i in range(l, r):
            arr[i] = aux[i]
            yield arr

    yield from _merge_sort(0, len(arr))
