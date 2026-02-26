"""Two pointers technique."""

# 1. one input, opposite ends

# def fn(arr):
#     left = ans = 0
#     right = len(arr) - 1
#
#     while left < right:
#         # do some logic here with left and right
#         if CONDITION:
#             left += 1
#         else:
#             right -= 1
#
#     return ans

# 2. two inputs, one pointer for each input or two inputs, exhaust both

# def fn(arr1, arr2):
#     i = j = ans = 0
#
#     while i < len(arr1) and j < len(arr2):
#         # do some logic here
#         if CONDITION:
#             i += 1
#         else:
#             j += 1
#
#     while i < len(arr1):
#         # do logic
#         i += 1
#
#     while j < len(arr2):
#         # do logic
#         j += 1
#
#     return ans
