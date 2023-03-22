#!/usr/bin/env python3
def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.

    Args:
        matrix: a nested list representing a matrix

    Returns:
        a list of integers representing the shape of the matrix
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
