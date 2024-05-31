#!/usr/bin/env python3
"""  that calculates the cofactor matrix of a matrix: """
minor = __import__('1-minor').minor


def cofactor(matrix):
    """
    Calculates the minor cofactor of a given matrix.

    Args:
        matrix (list of lists): The input matrix.

    Returns:
        the cofactor matrix of matrix
    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.

    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for i in matrix:
        if not isinstance(i, list):
            raise TypeError("matrix must be a list of lists")
    num_rows = len(matrix)
    for i in matrix:
        if num_rows != len(i):
            raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = minor(matrix)
    cofactor_matrix = []

    for i in range(len(minor_matrix)):
        row = []
        for j in range(len(minor_matrix[i])):
            row.append((-1) ** (i + j) * minor_matrix[i][j])
        cofactor_matrix.append(row)

    return cofactor_matrix
