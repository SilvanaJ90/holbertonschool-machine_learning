#!/usr/bin/env python3
"""
    Determines the steady state probabilities
    of a regular markov chain
"""
import numpy as np


def regular(P):
    """

    P is a is a square 2D numpy.ndarray of shape
    (n, n) representing the transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing
    the steady state probabilities, or None on failure


    """
    if not isinstance(P, np.ndarray):
        return None
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None

    n = P.shape[0]

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # Find the index of the eigenvalue equal to 1
    idx = np.where(np.isclose(eigenvalues, 1))[0]
    if len(idx) != 1:
        return None

    # Get the corresponding eigenvector
    steady_state = np.real_if_close(np.abs(eigenvectors[:, idx]).T)

    # Normalize the eigenvector
    steady_state /= np.sum(steady_state)

    return steady_state
