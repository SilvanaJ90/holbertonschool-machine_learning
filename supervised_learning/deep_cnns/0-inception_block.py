#!/usr/bin/env python3
"""This module contains the function inception_block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """This module contains the function inception_block.

    This function builds an Inception block as described in the paper
    "Going Deeper with Convolutions" by Szegedy et al. It concatenates
    multiple convolutional branches with different kernel sizes and
    max-pooling operations to produce a single output tensor.

    Args:
        A_prev: Output tensor from the previous layer.
        filters (tuple): Tuple containing F1, F3R, F3, F5R, F5, and FPP,
            respectively:
            - F1 (int): Number of filters in the 1x1 convolution.
            - F3R (int): Number of filters in the 1x1 convolution before
                the 3x3 convolution.
            - F3 (int): Number of filters in the 3x3 convolution.
            - F5R (int): Number of filters in the 1x1 convolution before
                the 5x5 convolution.
            - F5 (int): Number of filters in the 5x5 convolution.
            - FPP (int): Number of filters in the 1x1 convolution after
                the max pooling.

    Returns:
        tf.Tensor: Concatenated output tensor of the Inception block.
    """

    F1, F3R, F3, F5R, F5, FPP = filters

    conv1x1 = K.layers.Conv2D(
        F1, (1, 1),
        padding='same', activation='relu')(A_prev)

    conv3x3R = K.layers.Conv2D(
        F3R, (1, 1),
        padding='same', activation='relu')(A_prev)

    conv3x3 = K.layers.Conv2D(
        F3, (3, 3),
        padding='same', activation='relu')(conv3x3R)

    conv5x5R = K.layers.Conv2D(
        F5R, (1, 1),
        padding='same', activation='relu')(A_prev)

    conv5x5 = K.layers.Conv2D(
        F5, (5, 5),
        padding='same', activation='relu')(conv5x5R)

    max_pool = K.layers.MaxPool2D(
        pool_size=(3, 3), strides=(1, 1),
        padding='same')(A_prev)
    conv1x1_pp = K.layers.Conv2D(
        FPP, (1, 1),
        padding='same', activation='relu')(max_pool)

    return K.layers.Concatenate()(
        [conv1x1, conv3x3, conv5x5, conv1x1_pp])
