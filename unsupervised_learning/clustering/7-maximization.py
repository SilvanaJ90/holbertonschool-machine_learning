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
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None
    n, d = X.shape
    if type(g) is not np.ndarray or g.ndim != 2 or g.shape[1] != n \
       or not np.all((g >= 0) & (g <= 1)):
        return None, None, None

    k = g.shape[0]
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    nk = np.sum(g, axis=1)
    pi = nk / n

    for i in range(k):
        gi = g[i].reshape((-1, 1))
        m[i] = np.sum(gi * X, axis=0) / nk[i]
        Xm = X - m[i]
        S[i] = np.matmul(Xm.T, Xm * gi) / nk[i]
        return pi, m, S
