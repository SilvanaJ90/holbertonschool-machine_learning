#!/usr/bin/env python3
""" Doc """


class Poisson:
    """ Class Poisson """
    def __init__(self, data=None, lambtha=1.):
        self.data = data
        self.lambtha = lambtha
        
    def lambtha(self, data):
        lambtha = float(lambtha)
           
        if not data:
            data = lambtha
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            lambtha * data
            if data < 2:
                raise ValueError("data must contain multiple values")
        return lambtha