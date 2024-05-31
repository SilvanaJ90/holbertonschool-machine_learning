#!/usr/bin/env python3
""" calculates the cost of a neural network with L2 regularization """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
        cost is the cost of the network without L2 regularization
        lambtha is the regularization parameter
        weights is a dictionary of the weights and biases
        (numpy.ndarrays) of the neural network
        L is the number of layers in the neural network
        m is the number of data points used
        Returns:
            the cost of the network accounting for L2 regularization
    """
    reg_term = 0

    for i in range(1, L + 1):
        weights_i = 'W' + str(i)
        reg_term += np.linalg.norm(weights[weights_i])
    l2_cost = cost + (lambtha / (2 * m)) * reg_term
    return l2_cost
