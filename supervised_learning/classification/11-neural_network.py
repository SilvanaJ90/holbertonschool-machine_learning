#!/usr/bin/env python3
""" Doc """
import numpy as np


class NeuralNetwork:
    """ Class NeuralNetwork"""
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """ performs the forward propagation of the neural network. """
        self.__A1 = self.sigmoid(np.dot(self.W1, X) + self.b1)
        self.__A2 = self.sigmoid(np.dot(self.W2, self.A1) + self.b2)
        return self.A1, self.A2

    def sigmoid(self, Z):
        """ implements the sigmoid activation function """
        return 1 / (1 + np.exp(-Z))

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost
