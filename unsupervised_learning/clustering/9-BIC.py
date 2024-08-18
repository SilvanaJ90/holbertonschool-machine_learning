#!/usr/bin/env python3
""" Finds the best number of clusters for a GMM using Bayesian Information Criterion """
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization

def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using Bayesian Information Criterion

    Parameters:
    X (numpy.ndarray): The data set of shape (n, d).
    kmin (int): Minimum number of clusters to check.
    kmax (int): Maximum number of clusters to check. If None, set to max possible clusters.
    iterations (int): Maximum number of iterations for the EM algorithm.
    tol (float): Tolerance for convergence.
    verbose (bool): Whether to print EM algorithm details.

    Returns:
    best_k (int): Best number of clusters based on BIC.
    best_result (tuple): Tuple (pi, m, S) containing priors, means, and covariances for best k.
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

    n, d = X.shape
    kmax = kmax or n  # Set kmax to the number of data points if None

    best_k = None
    best_result = None
    l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)

    for k in range(kmin, kmax + 1):
        # Fit GMM using EM algorithm
        gmm = GaussianMixture(n_components=k, max_iter=iterations, tol=tol, verbose=verbose)
        gmm.fit(X)
        
        # Calculate log-likelihood
        log_likelihood = gmm.score(X) * n
        l[k - kmin] = log_likelihood
        
        # Calculate BIC
        p = k * (d + d * (d + 1) / 2) + k - 1  # Number of parameters: k*(d + d*(d + 1)/2) + k - 1
        bic = p * np.log(n) - 2 * log_likelihood
        b[k - kmin] = bic

        # Check for the best BIC
        if best_k is None or bic < b[best_k - kmin]:
            best_k = k
            best_result = (gmm.weights_, gmm.means_, gmm.covariances_)

    return best_k, best_result, l, b
