#!/usr/bin/env python3
"""
    Multiplication: two matrices element-wise
    Args:  mat1 and mat2 matrices containing ints/floats
    Returns: new matrix, If mat1 and mat2 are not the same shape, return None
"""
import numpy as np


def mat_mul(mat1, mat2):
    """   Multiplication: two matrices element-wise """
    if len(mat1[0]) != len(mat2):
        return None
    arr1 = np.array(mat1)
    arr2 = np.array(mat2)
    result = arr1 @ arr2
    return result.tolist()
