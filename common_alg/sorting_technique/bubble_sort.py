# Bubble Sort

from typing import Generator, List


def bubble_sort(a: List[int]) -> Generator[List[int], None, None]:
    '''
    Bubble Sort Algorithm
    :param a: List of integers to be sorted
    :yield: The list after each swap operation
    '''
    arr = a
    n = len(arr)
    for i in range(n):
        for j in range(0, n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
            yield arr
