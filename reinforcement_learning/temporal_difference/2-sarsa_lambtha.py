#!/usr/bin/env python3
""" that performs SARSA(λ) """
import numpy as np


def sarsa_lambtha(
        env, Q, lambtha, episodes=5000, max_steps=100,
        alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
        epsilon_decay=0.05):
    """
    - env is the openAI environment instance
    - Q is a numpy.ndarray of shape (s,a) containing the Q table
    - lambtha is the eligibility trace factor
    - episodes is the total number of episodes to train over
    - max_steps is the maximum number of steps per episode
    - alpha is the learning rate
    - gamma is the discount rate
    - epsilon is the initial threshold for epsilon greedy
    - min_epsilon is the minimum value that epsilon should decay to
    - epsilon_decay is the decay rate for updating epsilon between episodes
    Returns: Q, the updated Q table

    """
    states, actions = Q.shape

    max_epsilon = epsilon

    def epsilon_greedy(epsilon, Qs):
        p = np.random.uniform()
        if p > epsilon:
            return np.argmax(Qs)
        else:
            return np.random.randint(0, actions)

    for i in range(episodes):
        E = np.zeros((states, actions))
        s_prev = env.reset()
        action_prev = epsilon_greedy(epsilon, Q[s_prev])

        for j in range(max_steps):
            s, reward, done, info = env.step(action_prev)
            action = epsilon_greedy(epsilon, Q[s])
            delta = reward + (gamma * Q[s, action]) - Q[s_prev, action_prev]
            E[s_prev, action_prev] += 1
            E = E * gamma * lambtha
            Q = Q + (alpha * delta * E)

            if done:
                break

            s_prev = s
            action_prev = action

        epsilon = min_epsilon + (
            max_epsilon - min_epsilon) * np.exp(-epsilon_decay * i)

    return Q