#!/usr/bin/env python3
""" that calculates the probability density
    function of a Gaussian distribution
"""
import numpy as np


def pdf(X, m, S):
    """

    X is a numpy.ndarray of shape (n, d) containing the
    data points whose PDF should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the distribution
    S is a numpy.ndarray of shape (d, d) containing
    the covariance of the distribution
    You are not allowed to use any loops
    You are not allowed to use the function numpy.diag
    or the method numpy.ndarray.diagonal
    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the
        PDF values for each data point
    All values in P should have a minimum value of 1e-300

    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None
    d = X.shape[1]
    if type(m) is not np.ndarray or m.ndim != 1 or m.shape[0] != d:
        return None
    if type(S) is not np.ndarray or S.ndim != 2 or S.shape != (d, d):
        return None
    Xm = X - m
    e = - 0.5 * np.sum(Xm * np.matmul(np.linalg.inv(S), Xm.T).T, axis=1)
    num = np.exp(e)
    det = np.linalg.det(S)
    prob = num / np.sqrt(((2 * np.pi) ** d) * det)
    return np.maximum(prob, 1e-300)
