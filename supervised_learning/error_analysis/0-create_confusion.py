#!/usr/bin/env python3
""" doc """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ doc """
    confusion = np.dot(labels.T, logits)
    return confusion
