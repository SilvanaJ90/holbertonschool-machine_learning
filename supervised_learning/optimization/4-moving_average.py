#!/usr/bin/env python3
"""
    calculates the weighted moving average of a data set
"""
import numpy as np


def moving_average(data, beta):
    """"
    Arg:
        data is the list of data to calculate the moving average of
        beta is the weight used for the moving average
        Your moving average calculation should use bias correction
        Returns: a list containing the moving averages of data
    """
    avg = []
    prev = 0
    for i, j in enumerate(data):
        prev = (beta * prev + (1 - beta) * j)
        correction = prev / (1 - (beta ** (i + 1)))
        avg.append(correction)
    return avg
