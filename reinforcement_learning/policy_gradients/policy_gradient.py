#!/usr/bin/env python3
import numpy as np
import gym


def policy(matrix, weight):
    """
    Doc
    """
    z = matrix.dot(weight)
    exp = np.exp(z)
    return exp/np.sum(exp)


def softmax_grad(softmax):
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def policy_gradient(state, weight):
    """
    - state: matrix representing the current observation of the environment
    - weight: matrix of random weight
    Return: the action and the gradient (in this order)
    """
    probs = policy(state, weight)

    action = np.random.choice(len(probs[0]), p=probs[0])

    dsoftmax = softmax_grad(probs)[action, :]

    dlog = dsoftmax / probs[0, action]

    grad = state.T.dot(dlog[None, :])

    return action, grad
