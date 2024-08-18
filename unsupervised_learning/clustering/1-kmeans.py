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

    if not isinstance(X, np.ndarray) or not isinstance(k, int) or not isinstance(iterations, int):
        return None, None

    
    n, d = X.shape
    
    # Initialize centroids using a uniform distribution
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(min_vals, max_vals, size=(k, d))
    
    for _ in range(iterations):
        # Compute the distances from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        
        # Assign each point to the nearest centroid
        clss = np.argmin(distances, axis=1)
        
        # Save the old centroids for convergence check
        C_old = C.copy()
        
        # Update centroids
        for i in range(k):
            if np.any(clss == i):
                C[i] = X[clss == i].mean(axis=0)
            else:
                # Reinitialize centroid if no points are assigned to it
                C[i] = np.random.uniform(min_vals, max_vals, size=(d,))
        
        # Check if centroids have changed
        if np.all(C == C_old):
            break
    
    return C, clss