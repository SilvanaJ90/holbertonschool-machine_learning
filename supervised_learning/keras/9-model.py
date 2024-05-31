#!/usr/bin/env python3
"""saves an entire model"""
import tensorflow.keras as K


def save_model(network, filename):
    """

    network is the model to save
    filename is the path of the file that the model should be saved to
    Returns: None

    """
    network.save(filename)


def load_model(filename):
    """

    filename is the path of the file that the model should be loaded from
    Returns: the loaded model

    """
    return K.models.load_model(filename)
