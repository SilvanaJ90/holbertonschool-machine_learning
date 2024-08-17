#!/usr/bin/env python3
""" Doc """
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Baum-Welch algorithm for Hidden Markov Models
    """
    T = Observations.shape[0]  # Length of observation sequence
    M = Transition.shape[0]    # Number of hidden states
    N = Emission.shape[1]      # Number of output states
    
    # Helper functions
    def forward(Obs, A, B, Pi):
        alpha = np.zeros((T, M))
        alpha[0] = Pi.flatten() * B[:, Obs[0]]
        for t in range(1, T):
            for j in range(M):
                alpha[t, j] = np.sum(alpha[t - 1] * A[:, j]) * B[j, Obs[t]]
        return alpha

    def backward(Obs, A, B):
        beta = np.zeros((T, M))
        beta[-1] = np.ones(M)
        for t in range(T - 2, -1, -1):
            for i in range(M):
                beta[t, i] = np.sum(beta[t + 1] * A[i] * B[:, Obs[t + 1]])
        return beta

    def compute_gammas(alpha, beta):
        gammas = alpha * beta
        return gammas / np.sum(gammas, axis=1, keepdims=True)

    def compute_xi(alpha, beta, Obs, A, B):
        xi = np.zeros((T - 1, M, M))
        for t in range(T - 1):
            denom = np.sum(alpha[t, :, None] * beta[t + 1] * A * B[:, Obs[t + 1]], axis=1)
            for i in range(M):
                xi[t, i] = (alpha[t, i] * beta[t + 1] * A[i] * B[:, Obs[t + 1]]) / denom
        return xi

    # Expectation-Maximization
    for _ in range(iterations):
        alpha = forward(Observations, Transition, Emission, Initial)
        beta = backward(Observations, Transition, Emission)
        gammas = compute_gammas(alpha, beta)
        xi = compute_xi(alpha, beta, Observations, Transition, Emission)

        # Update Transition matrix
        Transition = np.sum(xi, axis=0) / np.sum(gammas[:-1], axis=0, keepdims=True)
        
        # Update Emission matrix
        new_Emission = np.zeros_like(Emission)
        for k in range(N):
            mask = (Observations == k)
            new_Emission[:, k] = np.sum(gammas[mask], axis=0)
        Emission = new_Emission / np.sum(gammas, axis=0, keepdims=True)
        
    return Transition, Emission
