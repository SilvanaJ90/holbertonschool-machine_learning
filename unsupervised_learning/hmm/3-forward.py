#!/usr/bin/env python3
""" Doc """
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ Doc """
    if not isinstance(
        Observation, np.ndarray) or not isinstance(
            Emission, np.ndarray) \
            or not isinstance(
                Transition, np.ndarray) or not isinstance(
                    Initial, np.ndarray):
        return None, None

    if len(Observation.shape) != 1 or len(
        Emission.shape) != 2 or len(Transition.shape) != 2 \
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

    # Initialize the forward path probabilities
    F = np.zeros((N, T))
    F[:, 0] = np.multiply(Initial[:, 0], Emission[:, Observation[0]])

    # Perform the forward algorithm
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(
                F[:, t - 1] * Transition[:, j] * Emission[j, Observation[t]])

    # Calculate the likelihood of the observations given the model
    P = np.sum(F[:, -1])

    return P, F
