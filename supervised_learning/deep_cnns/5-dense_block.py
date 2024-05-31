#!/usr/bin/env python3
"""  builds a dense block """
import tensorflow.keras as K


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
    he_normal = K.initializers.he_normal()
    concat_outputs = [X]
    for i in range(layers):
        # Bottleneck layer
        x = K.layers.BatchNormalization()(X)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(
            growth_rate * 4,
            (1, 1), padding='same', kernel_initializer=he_normal
        )(X)

        # Convolutional layer
        x = K.layers.BatchNormalization()(x)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(
            growth_rate, 3, padding='same',
            kernel_initializer=he_normal)(x)

        concat_outputs.append(x)
        X = K.layers.concatenate(concat_outputs, axis=-1)
        nb_filters += growth_rate

    return X, nb_filters
