#!/usr/bin/env python3
""" doc """
import numpy as np


def sensitivity(confusion):
    """ doc """
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1)
    sensitivity_value = TP/FN
    return sensitivity_value
