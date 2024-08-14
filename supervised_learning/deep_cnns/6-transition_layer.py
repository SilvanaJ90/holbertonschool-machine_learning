#!/usr/bin/env python3
"""  builds a dense block """
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer
    Your code should implement compression as used in DenseNet-C
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization and a
    rectified linear activation (ReLU), respectively
    Returns: The output of the transition layer and the number
    of filters within the output, respectively
    """
    initializer = K.initializers.he_normal(seed=0)

    # Batch Normalization + ReLU
    bn = K.layers.BatchNormalization()(X)
    relu = K.layers.Activation('relu')(bn)

    # 1x1 Convolution
    nb_filters = int(nb_filters * compression)
    conv = K.layers.Conv2D(
        nb_filters, (1, 1),
        padding='same', kernel_initializer=initializer)(relu)

    # Average Pooling (2x2)
    avg_pool = K.layers.AveragePooling2D(
        pool_size=(2, 2), strides=(2, 2), padding='same')(conv)

    return avg_pool, nb_filters
