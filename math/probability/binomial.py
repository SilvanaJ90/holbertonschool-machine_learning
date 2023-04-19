#!/usr/bin/env python3
""" Represents a binomial distribution """


class Binomial:
    """ class Binomial """
    def __init__(self, data=None, n=1, p=0.5):
        n = int(n)
        p = float(p)
        if (n < 0):
            raise ValueError("n must be a positive value")
        if (p no probabilidad):
            raise ValueError("p must be greater than 0 and less than 1")
        if isinstance(data, not list):
            raise TypeError("data must be a list")