#!/usr/bin/env python3
""" Doc """
import numpy as np


def kmeans(X, k, iterations=1000):
    """ Doc """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    C = np.random.uniform(min_vals, max_vals, size=(k, d))

    for i in range(iterations):

        centroids = np.copy(C)
        centroids_extended = C[:, np.newaxis]

        distances = np.sqrt(((X - centroids_extended) ** 2).sum(axis=2))

        clss = np.argmin(distances, axis=0)

        for c in range(k):
            if X[clss == c].size == 0:
                C[c] = np.random.uniform(min_vals, max_vals, size=(1, d))
            else:
                C[c] = X[clss == c].mean(axis=0)

        centroids_extended = C[:, np.newaxis]
        distances = np.sqrt(((X - centroids_extended) ** 2).sum(axis=2))
        clss = np.argmin(distances, axis=0)

        if (centroids == C).all():
            break

    return C, clss
