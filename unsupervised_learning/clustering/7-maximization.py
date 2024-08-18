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
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None
    
    n, d = X.shape
    k = g.shape[0]

    if g.shape != (k, n):
        return None, None, None

    # Calculate the total responsibilities for each cluster
    total_responsibilities = np.sum(g, axis=1)

    # Calculate the updated priors
    pi = total_responsibilities / n

    # Initialize m and S
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    # Calculate the updated means
    for i in range(k):
        weighted_sum = np.dot(g[i], X) / total_responsibilities[i]
        m[i] = weighted_sum

    # Calculate the updated covariances
    for i in range(k):
        diff = X - m[i]
        weighted_cov = np.dot(g[i] * diff.T, diff) / total_responsibilities[i]
        S[i] = weighted_cov

    return pi, m, S
