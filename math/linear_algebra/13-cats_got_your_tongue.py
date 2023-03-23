#!/usr/bin/env python3
"""
    Concatenates two matrices along a specific axis
    Args:  mat1 and mat2 matrices containing ints/floats
           and axis as an integer representing
           the axis along which the matrices should be concatenated.
    Returns: new matrix, if the two matrices cannot
             be concatenated, return None
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ Concatenates two matrices along a specific axis """
    return np.concatenate((mat1, mat2), axis=axis)
