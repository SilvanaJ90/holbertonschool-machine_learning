#!/usr/bin/env python3
""" that performs the expectation maximization for a GMM: """

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the expectation maximization for a GMM"""
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0 or type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0 or type(verbose) is not bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, log_likelihood = expectation(X, pi, m, S)

    for i in range(iterations):
        pi, m, S = maximization(X, g)
        g, new_log_likelihood = expectation(X, pi, m, S)

        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(f"Log Likelihood after {i} iterations: {log_likelihood:.5f}")

        if abs(new_log_likelihood - log_likelihood) <= tol:
            if verbose:
                print(f"Log Likelihood after {i + 1} iterations: {new_log_likelihood:.5f}")
            break

        log_likelihood = new_log_likelihood

    return pi, m, S, g, log_likelihood
