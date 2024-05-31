#!/usr/bin/env python3
""" calculates the F1 score of a confusion matrix """
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Arg:
    confusion is a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and
    column indices represent the predicted labels

    classes is the number of classes

    Returns: a numpy.ndarray of shape (classes,)
    containing the F1 score of each class
    """
    sens = sensitivity(confusion)
    prec = precision(confusion)
    f1_score = 2 * (prec * sens) / (prec + sens)
    return f1_score
