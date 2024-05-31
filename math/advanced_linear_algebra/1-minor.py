#!/usr/bin/env python3
""" calculates the minor matrix of a matrix: """
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """
    Calculates the minor matrix of a given matrix.

    Args:
        matrix (list of lists): The input matrix.

    Returns:
        list of lists: The minor matrix of the input matrix.

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

    minor_matrix = []
    for i in range(num_rows):
        minor_row = []
        for j in range(num_rows):
            minor = [
                row[:j] + row[j+1:]
                for row in (matrix[:i] + matrix[i+1:])]
            minor_row.append(determinant(minor))
        minor_matrix.append(minor_row)

    return minor_matrix
