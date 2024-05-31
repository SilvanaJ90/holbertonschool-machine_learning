#!/usr/bin/env python3
"""  calculates a correlation matrix: """
import numpy as np


def correlation(C):
    """ Doc """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]
    diag_sqrt = np.sqrt(np.diag(C))
    correlation_matrix = np.outer(diag_sqrt, diag_sqrt)
    correlation_matrix = C / correlation_matrix

    return correlation_matrix
