#!/usr/bin/env python3
"""
that builds the inception network
"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
        Builds the Inception network as described in

        The input data is assumed to have shape (224, 224, 3).
        Use a rectified linear activation (ReLU).

        Returns:
            keras.Model: The constructed Inception network model.

    """
    inputs = K.layers.Input(shape=(224, 224, 3))

    # First Convolutional layer
    conv1 = K.layers.Conv2D(
        64, (7, 7),
        strides=(2, 2), activation='relu', padding='same')(inputs)
    max_pool1 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(conv1)

    # Second Convolutional layer
    conv2_1 = K.layers.Conv2D(
        64, (1, 1), activation='relu', padding='same')(max_pool1)
    conv2_2 = K.layers.Conv2D(
        192, (3, 3), activation='relu', padding='same')(conv2_1)
    max_pool2 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(conv2_2)

    # Inception blocks
    inception_3a = inception_block(
        max_pool2, [64, 96, 128, 16, 32, 32])
    inception_3b = inception_block(
        inception_3a, [128, 128, 192, 32, 96, 64])
    max_pool3 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(inception_3b)

    inception_4a = inception_block(
        max_pool3, [192, 96, 208, 16, 48, 64])
    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64, 64])
    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64, 64])
    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64, 64])
    inception_4e = inception_block(inception_4d, [256, 160, 320, 32, 128, 128])
    max_pool4 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(inception_4e)

    inception_5a = inception_block(max_pool4, [256, 160, 320, 32, 128, 128])
    inception_5b = inception_block(inception_5a, [384, 192, 384, 48, 128, 128])

    # Average pooling and Dropout
    avg_pool = K.layers.AveragePooling2D(
        (7, 7), strides=(1, 1), padding='valid')(inception_5b)
    dropout = K.layers.Dropout(0.4)(avg_pool)

    # Output layer
    output = K.layers.Dense(1000, activation='softmax')(dropout)

    model = K.Model(inputs=inputs, outputs=output)

    return model
