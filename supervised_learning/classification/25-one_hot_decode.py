#!/usr/bin/env python3
"""" Doc """
import numpy as np


def one_hot_decode(one_hot):
    """ converts a one-hot matrix into a vector of labels:"""

    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    classes, m = one_hot.shape

    # Initialize an empty array for the tags
    labels = np.zeros(m, dtype=int)

    for i in range(m):
        index = np.argmax(one_hot[:, i])
        labels[i] = index

    return labels
