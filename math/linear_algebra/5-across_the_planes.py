#!/usr/bin/env python3
"""
    Adds two matrices element-wise
    Args:  mat1 and mat2 matrices containing ints/floats
    Returns: new matrix, If mat1 and mat2 are not the same shape, return None
"""


def add_matrices2D(mat1, mat2):
    """ Adds two matrices element-wise"""

    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    else:
        new_list = []
        for i in range(len(mat1)):
            row = []
            for j in range(len(mat1[0])):
                row.append(mat1[i][j] + mat2[i][j])
            new_list.append(row)
        return new_list
