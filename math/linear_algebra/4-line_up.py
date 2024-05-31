#!/usr/bin/env python3
"""
    Adds two arrays element-wise
    Args:  arr1 and arr2 are lists of ints/floats
    Returns: new list, if arr1 and arr2 are not the same shape, return None
"""


def add_arrays(arr1, arr2):
    """ Adds two arrays element-wise"""
    if len(arr1) != len(arr2):
        return None
    else:
        new_list = []
        for i in range(len(arr1)):
            new_list.append(arr1[i] + arr2[i])
        return new_list
