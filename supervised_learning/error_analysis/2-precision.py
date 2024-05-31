#!/usr/bin/env python3
""" doc """
import numpy as np


def precision(confusion):
    """ doc """
    TP = np.diag(confusion)
    PP = np.sum(confusion, axis=0)
    precision_value = TP/PP
    return precision_value
