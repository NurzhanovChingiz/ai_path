# Merge Sort

from collections.abc import Generator


def merge_sort(a: list[int]) -> Generator[list[int], None, None]:
    arr = a
    aux = [0] * len(arr)

    def _merge_sort(left: int, r: int) -> Generator[list[int], None, None]:
        if r - left <= 1:
            return
        m = (left + r) // 2
        yield from _merge_sort(left, m)
        yield from _merge_sort(m, r)
        i, j, k = left, m, left
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
        for i in range(left, r):
            arr[i] = aux[i]
            yield arr

    yield from _merge_sort(0, len(arr))
