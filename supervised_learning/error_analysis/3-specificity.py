#!/usr/bin/env python3
""" doc """
import numpy as np


def specificity(confusion):
    """ doc
      """
    TN = np.sum(confusion) - np.sum(confusion, axis=0) - np.sum(
        confusion, axis=1) + np.diag(confusion)
    FP = np.sum(confusion) - np.sum(confusion, axis=1)
    specificity_value = TN/FP
    return specificity_value
