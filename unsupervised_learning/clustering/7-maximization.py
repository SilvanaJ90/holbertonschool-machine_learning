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
        updated pi for each cluster
        m is a numpy.ndarray of shape (k, d) containing the updated
        centroid means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the updated
        covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    k = g.shape[0]
    n, d = X.shape

    nk = np.sum(g, axis=0)

    check = np.sum(nk)
    if check != X.shape[0]:
        return None, None, None

    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    pi =  np.zeros((k,))

    for i in range(k):
        mu_up = np.sum((g[i, :, np.newaxis] * X), axis=0)
        mu_down = np.sum(g[i], axis=0)
        m[i] = mu_up / mu_down

        x_m = X - m[i]
        sigma_up = np.matmul(g[i] * x_m.T, x_m)
        sigma_down = np.sum(g[i])
        S[i] = sigma_up / sigma_down

        pi[i] = np.sum(g[i]) / n
    return pi, m, S
