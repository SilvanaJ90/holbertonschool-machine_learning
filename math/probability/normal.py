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
        Calculates the z-score of a given x-value.

        :param x: The x-value to calculate the z-score for.
        :return: The z-score of x.
        """
        return (x - self.mu) / self.sigma

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score.

        :param z: The z-score to calculate the x-value for.
        :return: The x-value of z.
        """
        return z * self.sigma + self.mu
