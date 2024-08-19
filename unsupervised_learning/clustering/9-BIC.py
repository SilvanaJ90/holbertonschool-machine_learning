#!/usr/bin/env python3
""" Finds the best number of clusters for a
    GMM using Bayesian Information Criterion
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a
    GMM using Bayesian Information Criterion
    Parameters:
    X (numpy.ndarray): The data set of shape (n, d).
    kmin (int): Minimum number of clusters to check.
    kmax (int): Maximum number of clusters to check.
    If None, set to max possible clusters.
    iterations (int): Maximum number of iterations for the EM algorithm.
    tol (float): Tolerance for convergence.
    verbose (bool): Whether to print EM algorithm details.

    Returns:
    best_k (int): Best number of clusters based on BIC.
    best_result (tuple): Tuple (pi, m, S) containing priors,
    means, and covariances for best k.
    l (numpy.ndarray): Log likelihoods for each cluster size tested.
    b (numpy.ndarray): BIC values for each cluster size tested.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None
    if kmax <= kmin:
        return None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    best_k = None
    best_result = None
    log_likelihoods = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)

    # Loop through each number of clusters from kmin to kmax
    for k in range(kmin, kmax + 1):
        # Run expectation-maximization algorithm
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)

        # Compute BIC
        if pi is not None and m is not None \
           and S is not None and g is not None:
            n_params = k * (d + d * (d + 1) / 2) + k - 1
            BIC_value = n_params * np.log(n) - 2 * log_likelihood

            log_likelihoods[k - kmin] = log_likelihood
            b[k - kmin] = BIC_value

            # Update best k if current BIC is lower
            if best_k is None or BIC_value < b[best_k - kmin]:
                best_k = k
                best_result = (pi, m, S)

    return best_k, best_result, log_likelihoods, b
