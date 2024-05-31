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

        for i in range(1, self.__L + 1):
            Wl = self.__weights['W' + str(i)]
            bl = self.__weights['b' + str(i)]
            Al_prev = self.__cache['A' + str(i - 1)]

            Zl = np.dot(Wl, Al_prev) + bl
            Al = 1 / (1 + np.exp(-Zl))

            self.__cache['A' + str(i)] = Al

        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neurons predictions
        """
        A2 = self.forward_prop(X)[0]  # Use _ to discard the cache
        prediction = np.where(A2 >= 0.5, 1, 0)
        cost = self.cost(Y, A2)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        # total number of examples in the training data set.
        m = Y.shape[1]
        # calculating the error in the last layer of the neural network.
        dz_last = self.__cache['A' + str(self.__L)] - Y
        # This loop runs from the last layer (LL) to the first layer
        for i in range(self.__L, 0, -1):
            # Get the activations of the previous layer (A**i-1)
            A = self.__cache['A' + str(i - 1)]
            # Gradient with respect to the weights dw (dw=m1​⋅dzlast​⋅A**T)
            dw = np.dot(dz_last, A.T) / m
            # Gradient with respect to the bias (db = 1/m (∑ ​dzlast)
            db = np.sum(dz_last, axis=1, keepdims=True) / m
            # culate the new error (dz=WT⋅dzlast​⋅A⋅(1−A)
            dz = np.dot(self.__weights['W' + str(i)].T, dz_last) * A * (
                1 - A)

            self.__weights['W' + str(i)] -= alpha * dw
            self.__weights['b' + str(i)] -= alpha * db
            dz_last = dz

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network"""
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')

        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)

        A, cost = self.evaluate(X, Y)
        return A, cost
