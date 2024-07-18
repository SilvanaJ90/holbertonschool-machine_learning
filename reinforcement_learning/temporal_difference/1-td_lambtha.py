#!/usr/bin/env python3
"""  that performs the TD(λ) algorithm """
import numpy as np


def td_lambtha(
        env, V, policy, lambtha, episodes=5000,
        max_steps=100, alpha=0.1, gamma=0.99):
    """

    - env is the openAI environment instance
    - V is a numpy.ndarray of shape (s,) containing the value estimate
    - policy is a function that takes in a state and returns
        the next action to take
    - lambtha is the eligibility trace factor
    - episodes is the total number of episodes to train over
    - max_steps is the maximum number of steps per episode
    - alpha is the learning rate
    - gamma is the discount rate
    Returns: V, the updated value estimate

    """
    states = V.shape[0]

    for i in range(episodes):
        s = env.reset()
        E = np.zeros(states)
        for j in range(max_steps):
            action = policy(s)
            s_new, reward, done, info = env.step(action)

            delta = reward * (gamma * V[s_new]) - V[s]

            E *= gamma * lambtha
            E[s] += 1

            V = V + alpha * delta * E

            if done:
                break
            s = s_new
    return V
