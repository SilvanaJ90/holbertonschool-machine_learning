#!/usr/bin/env python3
"""  builds a dense block """
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """

    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block
    You should use the bottleneck layers used for DenseNet-B
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization and
    a rectified linear activation (ReLU), respectively
    Returns: The concatenated output of each layer within the Dense Block
    and the number of filters within the concatenated outputs, respectively


    """
    he_normal = K.initializers.he_normal(seed=0)

    for i in range(layers):
        # Batch Normalization + ReLU + 1x1 Convolution (Bottleneck layer)
        bn1 = K.layers.BatchNormalization()(X)
        relu1 = K.layers.Activation('relu')(bn1)
        conv1 = K.layers.Conv2D(
            4 * growth_rate, (1, 1),
            padding='same',
            kernel_initializer=he_normal)(relu1)

        # Batch Normalization + ReLU + 3x3 Convolution
        bn2 = K.layers.BatchNormalization()(conv1)
        relu2 = K.layers.Activation('relu')(bn2)
        conv2 = K.layers.Conv2D(
            growth_rate, (3, 3),
            padding='same',
            kernel_initializer=he_normal)(relu2)

        # Concatenate input with output of the 3x3 convolution
        X = K.layers.Concatenate()([X, conv2])

        # Update the number of filters
        nb_filters += growth_rate

    return X, nb_filters
