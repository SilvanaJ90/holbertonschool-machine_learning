#!/usr/bin/env python3
""" Tests for the optimum number of clusters by variance: """

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ DOc """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(kmin) is not int or kmin < 1:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax < 1:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None

    results = []
    var = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        var_value = variance(X, C)
        if var_value is None:
            return None, None
        results.append((C, clss))
        var.append(var_value)

    d0 = var[0]
    d_vars = []
    for v in var:
        d_vars.append(d0 - v)

    return results, d_vars
