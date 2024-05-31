#!/usr/bin/env python3
""" Doc """
import numpy as np


def initialize(X, k):
    """
    that initializes cluster centroids for K-means
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    centroids = np.random.uniform(min_vals, max_vals, size=(k, d))

    return centroids
