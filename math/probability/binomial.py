#!/usr/bin/env python3
""" Represents a binomial distribution """


class Binomial:
    """ class Binomial """
    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(round(n))
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            var = sum((x - mean) ** 2 for x in data) / len(data)
            self.p = 1 - var / mean
            self.n = round(mean ** 2 / (mean - var))
            self.p = mean / self.n

    def pmf(self, k):
        """ pmf """
        k = int(k)  # Convert k to an integer
        if k < 0 or k > self.n:
            return 0  # k is out of range
        else:
            # Calculate binomial probability mass function (PMF)
            pmf_value = (self.factorial(self.n) / (
                self.factorial(k) * self.factorial(
                    self.n - k))) * (self.p**k) * ((1 - self.p)**(self.n - k))
            return pmf_value

    def cdf(self, k):
        """ doc """
        k = int(k)  # Convert k to an integer
        if k < 0:
            return 0  # k is out of range
        else:
            cdf_value = sum(self.pmf(i) for i in range(k + 1))
            return cdf_value

    def factorial(self, num):
        """ Doc """
        result = 1
        for i in range(1, num + 1):
            result *= i
        return result
