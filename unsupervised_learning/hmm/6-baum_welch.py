#!/usr/bin/env python3
""" Doc """
forward = __import__('3-forward').forward
backward = __import__('5-backward').backward
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Baum-Welch algorithm for Hidden Markov Models
    """
    T = Observations.shape[0]
    N, M = Transition.shape

    for _ in range(iterations):
        # Forward Step
        forward_prob, _ = forward(Observations, Emission, Transition, Initial)

        # Backward Step
        _, backward_prob = backward(Observations, Emission, Transition, Initial)

        # Expectation step
        xi = np.zeros((T - 1, N, N))
        gamma = np.zeros((T, N))

        for t in range(T - 1):
            obs = Observations[t + 1]
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = forward_prob[i, t] * Transition[i, j] * Emission[j, obs] * backward_prob[j, t + 1]
            xi[t] /= np.sum(xi[t])

        gamma = np.sum(xi, axis=2)
        
        # Maximization step
        Transition_new = np.sum(xi, axis=0) / np.sum(gamma, axis=0).reshape((-1, 1))
        gamma_sum = np.sum(gamma, axis=0)
        gamma_sum = np.hstack((gamma_sum, np.sum(xi[-1], axis=0)))
        for k in range(M):
            Emission[k] = np.sum((Observations == k) * gamma.T, axis=0) / gamma_sum[k]

        # Check for convergence
        if np.allclose(Transition, Transition_new, atol=1e-8):
            return Transition, Emission

        Transition = Transition_new

    # If convergence is not reached
    return None, None