#!/usr/bin/env python3
""" Doc """


def summation_i_squared(n):
    """
        function Return sum
    """
    if n != int(n) or n < 1:
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6
