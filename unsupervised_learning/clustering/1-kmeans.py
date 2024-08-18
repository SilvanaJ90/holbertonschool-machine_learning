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
        centroids = np.copy(centroids)
        centroids_extended = centroids[:, np.newaxis]

        # distances also know as euclidean distance
        distances = np.sqrt(((X - centroids_extended) ** 2).sum(axis=2))
        # an array containing the index to the nearest centroid per each point
        clss = np.argmin(distances, axis=0)

        # Assign all points to the nearest centroid
        for c in range(k):
            if X[clss == c].size == 0:
                centroids[c] = np.random.uniform(min_vals, max_vals, size=(1, d))
            else:
                centroids[c] = X[clss == c].mean(axis=0)

        centroids_extended = centroids[:, np.newaxis]
        distances = np.sqrt(((X - centroids_extended) ** 2).sum(axis=2))
        clss = np.argmin(distances, axis=0)

        if (centroids == centroids).all():
            break

    return centroids, clss
