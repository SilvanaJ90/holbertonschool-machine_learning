#!/usr/bin/env python3
""" t builds the ResNet-50 architecture """

from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks should be followed
    by batch normalization along the channels axis and
    a rectified linear activation (ReLU), respectively.
    All weights should use he normal initialization
    Returns: the keras model
    """
    inputs = K.Input(shape=(224, 224, 3))
    he_normal_input = K.initializers.he_normal(seed=0)

    # Convolutional layer 1
    CV2D = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        padding='same',
        strides=(2, 2),
        kernel_initializer=he_normal_input)(inputs)

    # BatchNormalization
    BN = K.layers.BatchNormalization(axis=3)(CV2D)

    # Action
    AT = K.layers.ReLU()(BN)

    MX = K.layers.MaxPool2D(pool_size=(3, 3),
                            padding='same',
                            strides=(2, 2),)(AT)

    X = projection_block(MX, [64, 64, 256], s=1)
    for i in range(2):
        X = identity_block(X, [64, 64, 256])

    X = projection_block(X, [128, 128, 512])
    for i in range(3):
        X = identity_block(X, [128, 128, 512])

    X = projection_block(X, [256, 256, 1024])
    for i in range(5):
        X = identity_block(X, [256, 256, 1024])

    X = projection_block(X, [512, 512, 2048])
    for i in range(2):
        X = identity_block(X, [512, 512, 2048])

    X = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  padding='valid',
                                  strides=(1, 1))(X)

    # Output layer
    output = K.layers.Dense(1000, activation='softmax',
                            kernel_initializer=he_normal_input)(X)

    model = K.models.Model(inputs=inputs, outputs=output)

    return model
