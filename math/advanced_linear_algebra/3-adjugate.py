#!/usr/bin/env python3
"""  that calculates the adjugate matrix of a matrix: """
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """
    Calculates the adjugate of a given matrix.

    Args:
        matrix (list of lists): The input matrix.

    Returns:
        the adjugate matrix of matrix
    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.

    """
    cofactor_matrix = cofactor(matrix)
    num_rows = len(cofactor_matrix)
    num_cols = len(cofactor_matrix[0])

    adjugate_matrix = []

    for i in range(num_cols):
        row = []
        for j in range(num_rows):
            row.append(cofactor_matrix[j][i])
        adjugate_matrix.append(row)

    return adjugate_matrix
