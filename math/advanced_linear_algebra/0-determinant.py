#!/usr/bin/env python3
"""calculates the determinant of a matrix: """


def determinant(matrix):
    """
        matrix is a list of lists whose
        determinant should be calculated
        If matrix is not a list of lists,
        raise a TypeError with the message matrix must be a list of lists
        If matrix is not square, raise a ValueError
        with the message matrix must be a square matrix
        The list [[]] represents a 0x0 matrix
        Returns: the determinant of matrix

    """
    if not isinstance(
        matrix, list) or not all(isinstance(
            row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    for i in matrix:
        if len(i) != len(matrix):
            raise ValueError("matrix must be a square matrix")

    num_rows = len(matrix)

    # Base case: 0x0 matrix
    if num_rows == 0:
        return 1

    # Base case: 1x1 matrix
    if num_rows == 1:
        return matrix[0][0]

    # For a 2×2 Matrix
    if num_rows == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        det = (a * d) - (b * c)
        return det

    # For a 3×3 Matrix
    if num_rows == 3:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[0][2]
        d = matrix[1][0]
        e = matrix[1][1]
        f = matrix[1][2]
        g = matrix[2][0]
        h = matrix[2][1]
        i = matrix[2][2]
        det = (a * (e * i - f * h)) - \
              (b * (d * i - f * g)) + \
              (c * (d * h - e * g))
        return det

    # For larger matrices
    det = 0
    for j in range(num_rows):
        minor = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += (-1) ** j * matrix[0][j] * determinant(minor)

    return det
