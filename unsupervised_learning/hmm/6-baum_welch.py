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
    
    # Convert probabilities to log-space
    def log_add(a, b):
        return np.logaddexp(a, b)

    def forward(Obs, A, B, Pi):
        alpha = np.zeros((T, M))
        alpha[0] = np.log(Pi.flatten()) + np.log(B[:, Obs[0]])
        for t in range(1, T):
            for j in range(M):
                alpha[t, j] = np.log(np.sum(np.exp(alpha[t - 1] + np.log(A[:, j]))) + 1e-10) + np.log(B[j, Obs[t]])
        return alpha

    def backward(Obs, A, B):
        beta = np.zeros((T, M))
        beta[-1] = np.zeros(M)  # Log of 1
        for t in range(T - 2, -1, -1):
            for i in range(M):
                beta[t, i] = np.log(np.sum(np.exp(beta[t + 1] + np.log(A[i]) + np.log(B[:, Obs[t + 1]]))) + 1e-10)
        return beta

    def compute_gammas(alpha, beta):
        gammas = alpha + beta
        return gammas - np.log(np.sum(np.exp(gammas), axis=1, keepdims=True))

    def compute_xi(alpha, beta, Obs, A, B):
        xi = np.zeros((T - 1, M, M))
        for t in range(T - 1):
            denom = np.log(np.sum(np.exp(alpha[t][:, None] + np.log(A) + np.log(B[:, Obs[t + 1]]) + beta[t + 1]), axis=1) + 1e-10)
            for i in range(M):
                xi[t, i] = alpha[t, i][:, None] + np.log(A[i]) + np.log(B[:, Obs[t + 1]]) + beta[t + 1] - denom
        return np.exp(xi)

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
        row_sums = np.sum(new_Emission, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-10  # Avoid division by zero
        Emission = new_Emission / row_sums

    return Transition, Emission
