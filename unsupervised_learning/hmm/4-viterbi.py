#!/usr/bin/env python3
""" Doc """

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """ Doc """
    if not isinstance(Observation, np.ndarray) \
        or not isinstance(Emission, np.ndarray) \
            or not isinstance(Transition, np.ndarray) \
            or not isinstance(Initial, np.ndarray):
        return None, None

    if len(Observation.shape) != 1 or len(Emission.shape) != 2 \
        or len(Transition.shape) != 2 \
            or len(Initial.shape) != 2:
        return None, None

    if Emission.shape[0] != Transition.shape[0] \
        or Emission.shape[0] != Transition.shape[1] \
            or Transition.shape[0] != Transition.shape[1] \
            or Emission.shape[1] != Observation.max() + 1 \
            or Initial.shape[1] != 1:
        return None, None

    T = Observation.shape[0]
    N = Transition.shape[0]

    # Initialize the Viterbi path probabilities and backpointers
    V = np.zeros((N, T))
    backpointers = np.zeros((N, T), dtype=int)

    # Initialize the first column of V with Initial
    # probabilities and Emission probabilities
    V[:, 0] = np.multiply(Initial[:, 0], Emission[:, Observation[0]])

    # Iterate through the observations and compute
    # the Viterbi path probabilities
    for t in range(1, T):
        for j in range(N):
            # Compute the probabilities of transitioning
            # from previous states to the current state
            # and multiply by the emission probability of
            # the current observation given the current state
            temp_probs = V[:, t - 1] * Transition[:, j] \
                  * Emission[j, Observation[t]]
            # Choose the maximum probability and store it in V[j, t]
            V[j, t] = np.max(temp_probs)
            # Store the index of the previous
            # state that gives the maximum probability
            backpointers[j, t] = np.argmax(temp_probs)

    # Initialize the path with zeros
    path = np.zeros(T, dtype=int)

    # Backtrack through the backpointers to find the most likely path
    path[T - 1] = np.argmax(V[:, T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = backpointers[path[t + 1], t + 1]

    # Calculate the probability of obtaining the path sequence
    P = np.max(V[:, -1])

    return path.tolist(), P
