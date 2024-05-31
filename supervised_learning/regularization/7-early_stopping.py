#!/usr/bin/env python3
"""
Module to implement early stopping in gradient descent
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early

    Args:
        cost: current validation cost of the neural network
        opt_cost: lowest recorded validation cost of the neural network
        threshold: threshold used for early stopping
        patience: patience count used for early stopping
        count: count of how long the threshold has not been met

    Returns:
        a tuple with a boolean indicating whether to
        stop early and the updated count
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    stop_early = False
    if count >= patience:
        stop_early = True

    return stop_early, count
