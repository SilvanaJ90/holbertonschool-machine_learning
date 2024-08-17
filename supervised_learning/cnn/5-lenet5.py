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
    # Create a Sequential model
    model = K.Sequential()

    # Convolutional layer 1
    model.add(K.layers.Conv2D(
        filters=6, kernel_size=(5, 5),
        padding='same', activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0),
        input_shape=X.shape[1:]
    ))

    # Max pooling layer 1
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolutional layer 2
    model.add(K.layers.Conv2D(
        filters=16, kernel_size=(5, 5),
        padding='valid', activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    ))

    # Max pooling layer 2
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten the output from the previous layer
    model.add(K.layers.Flatten())

    # Fully connected layer 1
    model.add(K.layers.Dense(
        units=120, activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    ))

    # Fully connected layer 2
    model.add(K.layers.Dense(
        units=84, activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    ))

    # Output layer
    model.add(K.layers.Dense(
        units=10, activation='softmax'
    ))

    # Compile the model
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
