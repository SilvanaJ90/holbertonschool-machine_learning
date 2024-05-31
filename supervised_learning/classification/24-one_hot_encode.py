#!/usr/bin/env python3
"""" Doc """
import numpy as np


def one_hot_encode(Y, classes):
    """ Convert a numeric label vector into a one-hot matrix """

    if not isinstance(
        Y, np.ndarray) or len(Y) == 0 or not isinstance(
            classes, int) or max(Y) > classes:
        return None

    # Create a matrix of zeros with shape (len(Y), classes)
    one_hot_matrix = np.zeros((len(Y), classes))

    # Set the corresponding elements in each row to 1 based on the values in Y
    one_hot_matrix[np.arange(len(Y)), Y] = 1

    return one_hot_matrix.T
