#!/usr/bin/env python3
"""
     that conducts forward propagation using Dropout:
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """


    X is a numpy.ndarray of shape (nx, m) containing
    the input data for the network
        nx is the number of input features
        m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    L the number of layers in the network
    keep_prob is the probability that a node will be kept
    All layers except the last should use the tanh activation function
    The last layer should use the softmax activation function
    Returns: a dictionary containing the outputs of each layer
    and the dropout mask used on each layer (see example for format)

    """
    cache = {'A0': X}

    for layer in range(L):
        A = cache['A' + str(layer)]
        W = weights['W' + str(layer + 1)]
        b = weights['b' + str(layer + 1)]
        Z = np.matmul(W, A) + b

        if layer == L - 1:
            cache['A' + str(layer + 1)] = (
                np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True))
        else:
            cache['A' + str(layer + 1)] = np.tanh(Z)

            cache['D' + str(layer + 1)] = np.random.binomial(
                1, keep_prob, size=Z.shape
            )
            cache['A' + str(layer + 1)] *= cache['D' + str(
                layer + 1)] / keep_prob

    return cache
