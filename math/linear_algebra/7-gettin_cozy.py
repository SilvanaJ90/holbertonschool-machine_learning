#!/usr/bin/env python3
"""
    Concatenates two matrices along a specific axis
    Args:  mat1 and mat2 matrices containing ints/floats
           and axis as an integer representing
           the axis along which the matrices should be concatenated.
    Returns: new matrix, if the two matrices cannot
             be concatenated, return None
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenates two matrices along a specific axis """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        else:
            return [row for row in mat1] + [row for row in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        else:
            return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None
