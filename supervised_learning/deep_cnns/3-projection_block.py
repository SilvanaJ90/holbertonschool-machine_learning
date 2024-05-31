#!/usr/bin/env python3
""" builds a projection block """
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """

    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution as
        well as the 1x1 convolution in the shortcut connection
    s is the stride of the first convolution in both the main
    path and the shortcut connection
    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear
    activation (ReLU), respectively.
    All weights should use he normal initialization
    Returns: the activated output of the projection block

    """
    F11, F3, F12 = filters
    he_normal = K.initializers.he_normal()

    # Convolutional layer 1
    conv1 = K.layers.Conv2D(
        F11, (1, 1), strides=(s, s),
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

    # Shortcut connection
    shortcut = K.layers.Conv2D(
        F12, (1, 1),
        strides=(s, s),
        padding='valid',
        kernel_initializer=he_normal)(A_prev)
    shortcut_batch_norm = K.layers.BatchNormalization(axis=3)(shortcut)

    # Add shortcut and main path
    add = K.layers.Add()([batch_norm3, shortcut_batch_norm])

    # Activation function
    output = K.layers.Activation('relu')(add)

    return output
