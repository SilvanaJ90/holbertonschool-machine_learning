#!/usr/bin/env python3
"""
Module to create a layer with dropout in a neural network
"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function that should be used on the layer
        keep_prob: probability that a node will be kept

    Returns:
        the output of the new layer
    """

    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg')
    dropout = tf.layers.Dropout(rate=keep_prob)
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,

        name='layer')

    return dropout(layer(prev))
