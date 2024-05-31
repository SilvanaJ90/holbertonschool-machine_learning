#!/usr/bin/env python3
"""   that calculates the inverse of a matrix:: """
cofactor = __import__('2-cofactor').cofactor
determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """
    Calculates the inverse of a given matrix.

    Args:
        matrix (list of lists): The input matrix.

    Returns:
        the inverse matrix of matrix
    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.

    """

    adjugate_matrix = adjugate(matrix)
    determinant_matrix = determinant(matrix)

    if determinant_matrix == 0:
        return None

    num_rows = len(adjugate_matrix)
    num_cols = len(adjugate_matrix[0])

    inverse_matrix = []

    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            element = adjugate_matrix[i][j] / determinant_matrix
            row.append(element)
        inverse_matrix.append(row)

    return inverse_matrix
