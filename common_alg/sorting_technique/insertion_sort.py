# Insertion Sort

from typing import List, Generator


def insertion_sort(a: List[int]) -> Generator[List[int], None, None]:
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
