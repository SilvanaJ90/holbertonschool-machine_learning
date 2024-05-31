#!/usr/bin/env python3
"""   that calculates the definiteness of a matrix:: """
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a given matrix.

    Args:
        matrix (list of lists): The input matrix.

    Returns:
        the definiteness matrix of matrix
    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.

    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.allclose(matrix, matrix.T):
        return None  # Not symmetric

    eigenvalues, _ = np.linalg.eig(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
