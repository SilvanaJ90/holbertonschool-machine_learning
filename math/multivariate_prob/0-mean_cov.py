#!/usr/bin/env python3
""" that calculates the mean and covariance of a data set: """
import numpy as np


def mean_cov(X):
    """
    Doc
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    n, d = X.shape
    mean = np.mean(X, axis=0, keepdims=True)
    centered_data = X - mean
    cov = np.dot(centered_data.T, centered_data) / (n - 1)

    return mean, cov
