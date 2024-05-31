#!/usr/bin/env python3
""" Doc """

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Perform the backward algorithm for a hidden Markov model."""
    if (not isinstance(Observation, np.ndarray) or
            not isinstance(Emission, np.ndarray) or
            not isinstance(Transition, np.ndarray) or
            not isinstance(Initial, np.ndarray)):
        return None, None

    if (len(Observation.shape) != 1 or
            len(Emission.shape) != 2 or
            len(Transition.shape) != 2 or
            len(Initial.shape) != 2):
        return None, None

    if (Emission.shape[0] != Transition.shape[0] or
            Emission.shape[0] != Transition.shape[1] or
            Transition.shape[0] != Transition.shape[1] or
            Emission.shape[1] != Observation.max() + 1 or
            Initial.shape[1] != 1):
        return None, None

    T = Observation.shape[0]
    N = Transition.shape[0]

    # Initialize the backward path probabilities
    B = np.zeros((N, T))

    # Initialize the last column of B with 1's
    B[:, T - 1] = 1

    # Iterate through the observations in reverse order
    for t in range(T - 2, -1, -1):
        for i in range(N):
            # Compute the backward probability for state i at time t
            B[i, t] = np.sum(B[:, t + 1] * Transition[
                i, :] * Emission[:, Observation[t + 1]])

    # Calculate the likelihood of the observations given the model
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
