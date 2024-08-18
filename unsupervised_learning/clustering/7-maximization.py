#!/usr/bin/env python3
"""  that calculates the maximization step in the EM algorithm for a GMM:
"""
import numpy as np


def maximization(X, g):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster
    You may use at most 1 loop
    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the
        updated priors for each cluster
        m is a numpy.ndarray of shape (k, d) containing the updated
        centroid means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the updated
        covariance matrices for each cluster
    """
    try:
        # Validate input shapes
        if len(X.shape) != 2 or len(g.shape) != 2:
            return None, None, None
        
        n, d = X.shape
        k, n_ = g.shape
        
        if n != n_:
            return None, None, None
        
        # Calculate the sum of the posterior probabilities for each cluster
        Nk = np.sum(g, axis=1)  # Shape (k,)
        
        # Calculate the updated priors (pi)
        pi = Nk / n  # Shape (k,)
        
        # Calculate the updated means (m)
        m = (g @ X) / Nk[:, np.newaxis]  # Shape (k, d)
        
        # Calculate the updated covariance matrices (S)
        S = np.zeros((k, d, d))  # Initialize S with zeros
        for i in range(k):
            X_centered = X - m[i]  # Center the data for the i-th cluster
            S[i] = (g[i][:, np.newaxis] * X_centered).T @ X_centered / Nk[i]
        
        return pi, m, S
    
    except Exception:
        return None, None, None