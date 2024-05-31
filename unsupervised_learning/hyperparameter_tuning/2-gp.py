#!/usr/bin/env python3
""" Class GaussianProcess"""

import numpy as np


class GaussianProcess:
    """ Class GaussianProcess"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ that represents a noiseless 1D Gaussian process: """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """ Calculates the covariance kernel matrix between two matrices """
        k = (self.sigma_f**2) * np.exp(
            np.square(X1 - X2.T) / - (2 * (self.l ** 2)))
        return k

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation
        of points in a Gaussian process.

        X_s: numpy.ndarray of shape (s, 1) containing all of the points
        whose mean and standard deviation should be calculated.
        s: is the number of sample points.

        Returns:
            mu: numpy.ndarray of shape (s,) containing the
            mean for each point in X_s.
            sigma: numpy.ndarray of shape (s,) containing the
            variance for each point in X_s.
        """
        K_s = self.kernel(self.X, X_s)
        K_inv = np.linalg.inv(self.K)
        mu = np.matmul(np.matmul(K_s.T, K_inv), self.Y).reshape(-1)
        sigma = self.sigma_f**2 - np.sum(
            np.matmul(K_s.T, K_inv).T * K_s, axis=0)
        return mu, sigma

    def update(self, X_new, Y_new):
        """
        That updates a Gaussian Process:
        X_new is a numpy.ndarray of shape (1,)
        that represents the new sample point
        Y_new is a numpy.ndarray of shape (1,)
        that represents the new sample function value
        Updates the public instance attributes X, Y, and K
        """

        self.X = np.row_stack((self.X, X_new))
        self.Y = np.row_stack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)
