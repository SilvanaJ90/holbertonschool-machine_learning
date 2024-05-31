#!/usr/bin/env python3
""" That performs PCA on a dataset: """
import numpy as np


def pca(X, ndim):
    """ Doc """
    X = X - np.mean(X, axis=0)
    _, _, Vh = np.linalg.svd(X)
    W = Vh[:ndim].T
    return np.matmul(X, W)
