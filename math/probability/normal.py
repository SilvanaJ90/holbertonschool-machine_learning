#!/usr/bin/env python3
""" Doc """


class Normal:
    """ Class Normal """
    def __init__(self, data=None, mean=0., stddev=1.):
        self.mean = float(mean)
        self.stddev = float(stddev)

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = (sum((x - self.mean)**2
                               for x in data) / len(data))**(1/2)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        x is the x-value
        Returns the z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        z is the z-score
        Returns the x-value of z
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        x is the x-value
        Returns the PDF value for x
        """
        e = 2.7182818285
        π = 3.1415926536
        coef = 1 / (self.stddev * (2 * π) ** 0.5)
        exp = -(x - self.mean) ** 2 / (2 * self.stddev ** 2)
        return coef * e ** exp
    
    def cdf(self, x):
        import numpy as np
        """ Calculates the value of the CDF """
        t = 1 / (1 + 0.2316419 * (x - self.mean) / self.stddev)
        cdf_value = 1 - 0.3989423 * np.exp(-((x - self.mean) / self.stddev) ** 2 / 2) * (
                t * (0.319381530 - t * (0.356563782 - t * (1.781477937 - t * (1.821255978 - 0.379473726 * t)))))

        return cdf_value