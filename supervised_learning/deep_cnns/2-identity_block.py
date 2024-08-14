#!/usr/bin/env python3
""" builds an identity block """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block

    Args:
        A_prev (tensor): Output from the previous layer.
        filters (tuple or list): Contains F11, F3, F12, respectively:
            F11 (int): Number of filters in the first 1x1 convolution.
            F3 (int): Number of filters in the 3x3 convolution.
            F12 (int): Number of filters in the second 1x1 convolution.

    Returns:
        tensor: The activated output of the identity block.
    """
    F11, F3, F12 = filters
    he_normal = K.initializers.he_normal()

    # Convolutional layer 1
    conv1 = K.layers.Conv2D(
        F11, (1, 1),
        kernel_initializer=he_normal)(A_prev)
    batch_norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    activation1 = K.layers.Activation('relu')(batch_norm1)

    # Convolutional layer 2
    conv2 = K.layers.Conv2D(
        F3, (3, 3),
        padding='same',
        kernel_initializer=he_normal)(activation1)
    batch_norm2 = K.layers.BatchNormalization(axis=3)(conv2)
    activation2 = K.layers.Activation('relu')(batch_norm2)

    # Convolutional layer 3
    conv3 = K.layers.Conv2D(
        F12, (1, 1),
        padding='valid',
        kernel_initializer=he_normal)(activation2)
    batch_norm3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Add shortcut and main path
    add = K.layers.Add()([batch_norm3, A_prev])

    # Activation function
    output = K.layers.Activation('relu')(add)

    return output
