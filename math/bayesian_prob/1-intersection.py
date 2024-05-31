#!/usr/bin/env python3
""" that calculates the likelihood of obtaining this data given various
 hypothetical probabilities of developing severe side effects:
"""
import numpy as np
from scipy.special import comb


def likelihood(x, n, P):
    """
    Arg:
        -x is the number of patients that develop severe side effects
        - n is the total number of patients observed
        - P is a 1D numpy.ndarray containing the various
         hypothetical probabilities of developing severe side effects
    -Conditions:
        - If n is not a positive integer, raise a ValueError
         with the message n must be a positive integer
        - If x is not an integer that is greater than or
        equal to 0, raise a ValueError with the message x
        must be an integer that is greater than or equal to 0
        - If x is greater than n, raise a ValueError with the
         message x cannot be greater than n
        - If P is not a 1D numpy.ndarray, raise a TypeError
         with the message P must be a 1D numpy.ndarray
        - If any value in P is not in the range [0, 1], raise a
        ValueError with the message All values in P must be in the range [0, 1]
    Returns: a 1D numpy.ndarray containing the likelihood of obtaining the data
    x and n, for each probability in P, respectively
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if (P < 0).any() or (P > 1).any():
        raise ValueError("All values in P must be in the range [0, 1]")

    cb = comb(n, x)
    prob = cb * P**x * (1 - P)**(n - x)

    return prob


def intersection(x, n, P, Pr):
    """
    DOc
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if (P < 0).any() or (P > 1).any():
        raise ValueError("All values in P must be in the range [0, 1]")
    if (Pr < 0).any() or (Pr > 1).any():
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    intersections = likelihood(x, n, P) * Pr
    return intersections
