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

    def evaluate(self, X, Y):
        """
        Evaluates the neurons predictions
        """
        A1, A2 = self.forward_prop(X)
        prediction = np.where(A2 >= 0.5, 1, 0)
        cost = self.cost(Y, A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = X.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.sum(dz2) / m
        dz1 = np.matmul(self.W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1) / m

        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1
