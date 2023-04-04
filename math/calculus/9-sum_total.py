#!/usr/bin/env python3


def summation_i_squared(n):
    if n != int(n) or n < 1:
        return None
    elif n == 1:
        return 1  # base case: sum of 1^2 is 1
    else:
        return n**2 + summation_i_squared(n-1)  