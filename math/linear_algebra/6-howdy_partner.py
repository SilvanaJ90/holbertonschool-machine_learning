#!/usr/bin/env python3
"""
    Concatenates two arrays
    Args:  arr1 and arr2 are lists of ints/floats
    Returns: new list
"""


def cat_arrays(arr1, arr2):
    """ Concatenates two arrays """
    new_list = []
    for element in arr1:
        new_list.append(element)
    for element in arr2:
        new_list.append(element)
    return new_list
