#!/usr/bin/env python3
""" Doc """
import numpy as np


class DeepNeuralNetwork:
    """
    defines a deep neural network performing binary classification:
    """
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(1, self.__L + 1):
            self.__weights['b' + str(i)] = np.zeros((layers[i - 1], 1))
            if i == 1:
                self.__weights['W' + str(i)] = np.random.randn(
                    layers[i - 1], nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W' + str(i)] = np.random.randn(
                    layers[i - 1], layers[i - 2]) * np.sqrt(2 / layers[i - 2])

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        self.__cache['A0'] = X

        for l in range(1, self.__L + 1):
            Wl = self.__weights['W' + str(l)]
            bl = self.__weights['b' + str(l)]
            Al_prev = self.__cache['A' + str(l - 1)]

            Zl = np.dot(Wl, Al_prev) + bl
            Al = 1 / (1 + np.exp(-Zl))

            self.__cache['A' + str(l)] = Al

        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost
