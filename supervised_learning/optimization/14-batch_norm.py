#!/usr/bin/env python3
"""
that creates a batch normalization layer for a neural network in tensorflow
"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Arg:
    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should
    be used on the output of the layer
    you should use the tf.keras.layers.Dense layer as the base layer
    with kernal initializer
    tf.keras.initializers.VarianceScaling(mode='fan_avg')
    your layer should incorporate two trainable parameters,
    gamma and beta, initialized as vectors of 1 and 0 respectively
    you should use an epsilon of 1e-8
    Returns: a tensor of the activated output for the layer


    """
    activa = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, activation=None,
                                  kernel_initializer=activa, name='layer')
    Z = layer(prev)
    mu, sigma_2 = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(initial_value=tf.constant(
        1.0, shape=[n]), name='gamma')
    beta = tf.Variable(initial_value=tf.constant(
        0.0, shape=[n]), name='beta')
    z_b_norm = tf.nn.batch_normalization(
        Z, mu,
        sigma_2,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-8)
    return activation(z_b_norm)
