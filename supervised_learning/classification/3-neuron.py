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
        Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
        A is a numpy.ndarray with shape (1, m) containing the activated output of the neuron for each example
        To avoid division by zero errors, please use 1.0000001 - A instead of 1 - A
        Returns the cost
        """
        m = Y.shape[1]
        cost = -(1 / m)  * np.sum(Y * np.log(A) + np.log(1.0000001 - A))
        return cost
