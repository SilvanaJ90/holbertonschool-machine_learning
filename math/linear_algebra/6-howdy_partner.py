#!/usr/bin/env python3
"""
    Concatenates two arrays
    Args:  arr1 and arr2 are lists of ints/floats
    Returns: new list
"""


def cat_arrays(arr1, arr2):
    """ Concatenates two arrays """
    new_list = []
    new_list.append(arr1 + arr2)
    return new_list
