from .bubble_sort import bubble_sort
from .insertion_sort import insertion_sort
from .merge_sort import merge_sort
from .quick_sort import quick_sort

if __name__ == "__main__":
    # Check sorting algorithms
    data = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", data)
    bubble_sort_result = bubble_sort(data)
    insertion_sort_result = insertion_sort(data)
    merge_sort_result = merge_sort(data)
    quick_sort_result = quick_sort(data)
    
    print("Bubble Sorted array:", list(bubble_sort_result)[-1])
    print("Insertion Sorted array:", list(insertion_sort_result)[-1])
    print("Merge Sorted array:", list(merge_sort_result)[-1])
    print("Quick Sorted array:", list(quick_sort_result)[-1])