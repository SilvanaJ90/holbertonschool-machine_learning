#!/usr/bin/env python3
"""  that builds a neural network with the Keras library: """
import tensorflow.keras as K


# Ahora puedes utilizar K en lugar de tensorflow.keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx is the number of input features to the network
    layers is a list containing the number
    of nodes in each layer of the network
    activations is a list containing the
    activation functions used for each layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout
    You are not allowed to use the Input class
    Returns: the keras model

    """
    model = K.Sequential()

    for i in range(len(layers)):
        layer_name = f'dense_{i}'
        if i == 0:
            layer_name = 'dense'

        model.add(K.layers.Dense(
            layers[i],
            input_shape=(nx,),
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha),
            name=layer_name
        ))

        if i < len(layers) - 1:
            dropout_name = f'dropout_{i}'
            if i == 0:
                dropout_name = 'dropout'

            model.add(K.layers.Dropout(1 - keep_prob, name=dropout_name))

    return model
