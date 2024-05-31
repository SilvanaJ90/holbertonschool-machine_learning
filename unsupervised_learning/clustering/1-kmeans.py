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

    # Inicializar los centroides de los clusters
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    centroids = np.random.uniform(min_vals, max_vals, size=(k, d))

    if centroids.shape != (k, d):
        return None, None

    for i in range(iterations):
        # Asignación de Puntos a los Centroides más Cercanos
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)

        # Actualización de los centroides
        C = np.zeros((k, d))
        for c in range(k):
            if np.sum(clss == c) > 0:
                C[c] = X[clss == c].mean(axis=0)
            else:
                C[c] = np.random.uniform(min_vals, max_vals, size=(1, d))

        # Comprobación de Convergencia
        if np.array_equal(centroids, C):
            break
        centroids = np.copy(C)

    return centroids, clss
