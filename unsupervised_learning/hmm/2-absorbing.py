#!/usr/bin/env python3
"""
    Determines if a markov chain is absorbing
"""
import numpy as np


def absorbing(P):
    """

    P is a is a square 2D numpy.ndarray of shape (n, n)
    representing the standard transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure

    """
    if not isinstance(P, np.ndarray):
        return False
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return False

    n = P.shape[0]

    # Check if the matrix has absorbing states
    absorbing_states = np.diag(P) == 1
    if not any(absorbing_states):
        return False

    # Check if all other states are transient
    transient_states = np.logical_not(absorbing_states)
    for i in range(n):
        if transient_states[i]:
            if np.all(P[i, :] == 0):
                return False
            if np.all(P[:, i] == 0):
                return False

    return True
