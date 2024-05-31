#!/usr/bin/env python3
""" t builds the ResNet-50 architecture """
import tensorflow.keras as K

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
    inputs = K.layers.Input(shape=(224, 224, 3))
    he_normal = K.initializers.he_normal()

    # Convolutional layer 1
    X = K.layers.Conv2D(
        64, (7, 7),
        kernel_initializer=he_normal)(inputs)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = projection_block(X, [64, 64, 256], s=1)
    for i in range(2):
        X = identity_block(X, [64, 64, 256])

    X = projection_block(X, [128, 128, 512], s=1)
    for i in range(3):
        X = identity_block(X, [128, 128, 512])

    X = projection_block(X, [256, 256, 1024], s=1)
    for i in range(5):
        X = identity_block(X, [256, 256, 1024])

    X = projection_block(X, [512, 512, 2048], s=1)
    for i in range(2):
        X = identity_block(X, [512, 512, 2048])

    X = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  strides=(1, 1))(X)

    # Output layer
    output = K.layers.Dense(1000, activation='softmax',
                            kernel_initializer=he_normal)(X)

    model = K.Model(inputs=inputs, outputs=output)

    return model
