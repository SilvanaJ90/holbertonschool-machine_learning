#!/usr/bin/env python3
""" Doc """
import numpy as np


class Neuron:
    """ Class Neuron """

    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if (nx < 1):
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """ def forward"""
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]

        cost = -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        """
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)

        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        """
        m = Y.shape[1]

        dZ = A - Y
        dW = (1 / m) * np.dot(X, dZ.T)
        db = (1 / m) * np.sum(dZ)

        self.__W -= alpha * dW.T
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron
        X is a numpy.ndarray with shape (nx, m)
        that contains the input data

        nx is the number of input features to the neuron
        m is the number of examples

        Y is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data

        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

            if i % 100 == 0:
                cost = self.cost(Y, A)

        prediction, cost = self.evaluate(X, Y)
        return prediction, cost
