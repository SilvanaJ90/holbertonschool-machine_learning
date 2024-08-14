#!/usr/bin/env python3
""" builds the DenseNet-121 architecture """
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    growth_rate is the growth rate
    compression is the compression factor
    You can assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU), respectively
    All weights should use he normal initialization
    Returns: the keras model

    """
    inputs = K.layers.Input(shape=(224, 224, 3))
    he_normal_input = K.initializers.HeNormal(seed=0)
    nb_filters = 2 * growth_rate

    # Initial Convolution and Pooling layers
    x = K.layers.BatchNormalization(axis=3)(inputs)
    x = K.layers.ReLU()(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(
        nb_filters, (7, 7), strides=2,
        padding='same', kernel_initializer=he_normal_input)(x)
    x = K.layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Dense Block 1
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 6)

    # Transition Layer 1
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # Dense Block 2
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 12)

    # Transition Layer 2
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # Dense Block 3
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 24)

    # Transition Layer 3
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # Dense Block 4
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 16)

    x = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  padding='valid',
                                  strides=(1, 1))(x)
    # Fully connected layer (Classification layer)
    outputs = K.layers.Dense(
        1000, activation='softmax',
        kernel_initializer=he_normal_input)(x)

    # Create model
    model = K.models.Model(inputs=inputs, outputs=outputs)

    return model
