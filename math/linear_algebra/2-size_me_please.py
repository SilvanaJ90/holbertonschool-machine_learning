#!/usr/bin/env python3
import numpy as np
def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.

    Args:
        matrix: a nested list representing a matrix

    Returns:
        form -- tuple containing the number of rows and columns of the matrix
    """
    shape = np.shape(matrix)
    return shape
