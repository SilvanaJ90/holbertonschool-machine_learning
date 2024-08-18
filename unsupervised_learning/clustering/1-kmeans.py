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

    # Inicializar los centroides de los clusters
    n, d = X.shape

    # Initialize centroids using a multivariate uniform distribution
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    C = np.random.uniform(min_values, max_values, (k, d))

    clss = np.zeros(n)

    for _ in range(iterations):
        # Calculate distances and assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        new_clss = np.argmin(distances, axis=1)

        # If no change in clusters, return the result
        if np.all(new_clss == clss):
            break
        clss = new_clss

        # Update centroids
        for i in range(k):
            points_in_cluster = X[clss == i]
            if len(points_in_cluster) == 0:
                # Reinitialize the centroid if no points are assigned to the cluster
                C[i] = np.random.uniform(min_values, max_values, d)
            else:
                C[i] = np.mean(points_in_cluster, axis=0)

    return C, clss