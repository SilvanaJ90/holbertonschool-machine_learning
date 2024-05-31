#!/usr/bin/env python3
""" saves a model’s weights"""
import tensorflow.keras as K


def save_config(network, filename):
    """

    network is the model whose configuration should be saved
    filename is the path of the file that the configuration should be saved to
    Returns: None

    """
    with open(filename, 'w') as f:
        f.write(network.to_json())


def load_config(filename):
    """

    filename is the path of the file containing
    the model’s configuration in JSON format
    Returns: the loaded model


    """
    with open(filename, 'r') as f:
        return K.models.model_from_json(f.read())
