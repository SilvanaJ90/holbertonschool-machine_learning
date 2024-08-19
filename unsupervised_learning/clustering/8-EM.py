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
    # Initialize parameters
    pi, m, S = initialize(X, k)
    
    log_likelihood_old = None
    
    for i in range(iterations):
        # Expectation step
        g, _ = expectation(X, pi, m, S)
        
        # Maximization step
        pi, m, S = maximization(X, g)
        
        # Compute log likelihood
        log_likelihood_new = np.sum(np.log(np.sum(g, axis=0)))
        
        if log_likelihood_old is not None and np.abs(log_likelihood_new - log_likelihood_old) <= tol:
            if verbose:
                print(f"Log Likelihood after {i} iterations: {log_likelihood_new:.5f}")
            break
        
        log_likelihood_old = log_likelihood_new
        
        # Print log likelihood every 10 iterations and last iteration if verbose
        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(f"Log Likelihood after {i} iterations: {log_likelihood_new:.5f}")
    
    return pi, m, S, g, log_likelihood_new