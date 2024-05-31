#!/usr/bin/env python3
""" Represents a Multivariate Normal distribution """

import numpy as np
import math


class MultiNormal:
    """ Represents a Multivariate Normal distribution """
    def __init__(self, data):
        """ Initializes the MultiNormal instance """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1).reshape(-1, 1)
        self.cov = np.dot(data - self.mean, (
            data - self.mean).T) / (data.shape[1] - 1)

    @staticmethod
    def mean_cov(X):
        """calculates the mean and covariance of a data set

        Params:
            X: a numpy.ndarray of shape (n, d) containing the data set:
                n: the number of data points
                d: the number of dimensions in each data point

        Returns: mean, cov:
            mean: a numpy.ndarray of shape (1, d) containing
                the mean of the data set
            cov: a numpy.ndarray of shape (d, d) containing
                the covariance matrix of the data set
        """

        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            raise TypeError('X must be a 2D numpy.ndarray')

        n = X.shape[0]

        if n < 2:
            raise ValueError('X must contain multiple data points')

        mean = np.sum(X, axis=0) / n

        Z = (X - mean)
        cov = np.matmul(Z.T, Z) / (n - 1)

        return mean.reshape(-1, 1), cov

    def pdf(self, x):
        """calculates the PDF at a data point

        Params:
            x: a numpy.ndarray of shape (d, 1) containing the data point
               whose PDF should be calculated
                - d is the number of dimensions of the Multinomial instance

        Returns the value of the PDF
        """
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')

        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        right = (x - self.mean).T @ np.linalg.inv(self.cov) @ (x - self.mean)
        right = np.exp(-0.5 * right)

        left = 1 / (np.sqrt((2 * math.pi) **
                    self.mean.shape[0]) * np.sqrt(np.linalg.det(self.cov)))

        return (left * right)[0][0]
