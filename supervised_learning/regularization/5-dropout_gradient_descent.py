#!/usr/bin/env python3
""" hat updates the weights of a neural network
with Dropout regularization using gradient descent:"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """

        Y is a one-hot numpy.ndarray of shape (classes, m)
        that contains the correct labels for the data
            classes is the number of classes
            m is the number of data points
        weights is a dictionary of the weights and biases of the neural network
        cache is a dictionary of the outputs and dropout
        masks of each layer of the neural network
        alpha is the learning rate
        keep_prob is the probability that a node will be kept
        L is the number of layers of the network
        All layers use thetanh activation function except the last,
        which uses the softmax activation function
        The weights of the network should be updated in place

    """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache["A" + str(layer - 1)]
        W = weights["W" + str(layer)]
        b = weights["b" + str(layer)]

        dA = np.matmul(W.T, dZ)
        if layer > 1:
            dA *= cache["D" + str(layer - 1)]
        dA /= keep_prob

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        dZ = dA * (1 - np.power(A_prev, 2))

        weights["W" + str(layer)] -= alpha * dW
        weights["b" + str(layer)] -= alpha * db
