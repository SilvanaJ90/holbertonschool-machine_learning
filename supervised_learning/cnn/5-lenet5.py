#!/usr/bin/env python3
"""  hat builds a modified version of the LeNet-5 architecture using keras:
"""
from tensorflow import keras as K


def lenet5(X):
    """
        X is a K.Input of shape (m, 28, 28, 1) containing
        the input images for the network

        m is the number of images

    The model should consist of the following layers in order:

        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes

        All layers requiring initialization should initialize their
        kernels with the he_normal initialization method
        All hidden layers requiring activation should
        use the relu activation function
        you may import tensorflow.keras as K
    """
    initializer = K.initializers.he_normal()

    C1 = K.layers.Conv2D(filters=6,
                         kernel_size=(5, 5),
                         padding='same',
                         kernel_initializer=initializer,
                         activation=K.activations.relu)
    output_1 = C1(X)

    P2 = K.layers.MaxPool2D(pool_size=(2, 2),
                            strides=(2, 2))
    output_2 = P2(output_1)

    C3 = K.layers.Conv2D(filters=16,
                         kernel_size=(5, 5),
                         padding='valid',
                         kernel_initializer=initializer,
                         activation=K.activations.relu)
    output_3 = C3(output_2)

    P4 = K.layers.MaxPool2D(pool_size=(2, 2),
                            strides=(2, 2))
    output_4 = P4(output_3)

    output_5 = K.layers.Flatten()(output_4)

    FC6 = K.layers.Dense(units=120,
                         kernel_initializer=initializer,
                         activation=K.activations.relu)
    output_6 = FC6(output_5)

    FC7 = K.layers.Dense(units=84,
                         kernel_initializer=initializer,
                         activation=K.activations.relu)
    output_7 = FC7(output_6)

    FC8 = K.layers.Dense(units=10,
                         kernel_initializer=initializer)
    output_8 = FC8(output_7)

    softmax = K.layers.Softmax()(output_8)

    model = K.Model(inputs=X, outputs=softmax)

    optimizer = K.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model