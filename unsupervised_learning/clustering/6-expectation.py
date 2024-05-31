#!/usr/bin/env python3
""" that calculates the expectation step in the EM algorithm for a GMM:
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.

    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
    m is a numpy.ndarray of shape (k, d)
    containing the centroid means for each cluster
    S is a numpy.ndarray of shape (k, d, d)
    containing the covariance matrices for each cluster
    You may use at most 1 loop
    Returns: g, l, or None, None on failure
        g is a numpy.ndarray of shape (k, n) containing
        the posterior probabilities for each data point in each cluster
        l is the total log likelihood
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1 or \
       np.abs(np.sum(pi) - 1) > 1e-10:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d) or S.shape != (k, d, d):
        return None, None

    g = np.zeros((k, n))

    for i in range(k):
        likelihood = pdf(X, m[i], S[i])
        g[i] = pi[i] * likelihood

    # Normalize g
    g_sum = np.sum(g, axis=0)
    g /= g_sum

    # Calculate total log likelihood
    total_log_likelihood = np.sum(np.log(g_sum))

    return g, total_log_likelihood
