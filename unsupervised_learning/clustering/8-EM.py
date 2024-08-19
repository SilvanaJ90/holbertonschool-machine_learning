#!/usr/bin/env python3
"""Perform the Expectation-Maximization algorithm for Gaussian Mixture Models (GMM)."""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization

def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    doc
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) != int or k <= 0 or X.shape[0] < k:
        return None, None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) != float or tol < 0:
        return None, None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None, None
    # Initialize parameters
    pi, m, S = initialize(X, k)
    
    log_likelihood_old = 0
    i = 0
    
    for i in range(iterations):
        # Expectation step
        g, loglikelihood_new = expectation(X, pi, m, S)
        
        if verbose is True and (i % 10 == 0):
            print("Log Likelihood after {} iterations: {}".format(
                i, loglikelihood_new.round(5)))
        if abs(loglikelihood_new - log_likelihood_old) <= tol:
            break
        # Maximization step
        pi, m, S = maximization(X, g)
        i += 1
        log_likelihood_old = loglikelihood_new
    g, loglikelihood_new = expectation(X, pi, m, S)


    if verbose is True:
        print("Log Likelihood after {} iterations: {}".format(
            i, loglikelihood_new.round(5)))

    
    return pi, m, S, g, loglikelihood_new
