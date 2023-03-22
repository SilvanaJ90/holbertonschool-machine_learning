#!/usr/bin/env python3
"""
    Transpose of a 2D matrix, matrix
    Args: matrix: a nested transponse a matrix
    Returns: new matrix transponse
"""


def matrix_transpose(matrix):
    """ Transpose of a 2D matrix, matrix """
    row = len(matrix)
    columns = len(matrix[0])

    """Create an empty matrix with changed dimensions"""
    transposed_matrix = [[0 for j in range(row)] for i in range(columns)]

    """Transposed"""
    for i in range(row):
        for j in range(columns):
            transposed_matrix[j][i] = matrix[i][j]
    return transposed_matrix
