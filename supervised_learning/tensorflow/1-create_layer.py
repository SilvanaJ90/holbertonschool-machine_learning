#!/usr/bin/env python3
""" Function create_layer """
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


def create_layer(prev, n, activation):
    """
    prev is the tensor output of the previous layer
    n is the number of nodes in the layer to create
    activation is the activation function that the layer should use
    use tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    to implement He et. al initialization for the layer weights
    each layer should be given the name layer
    Returns: the tensor output of the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.keras.layers.Dense(
        units=n, activation=activation, kernel_initializer=init, name='layer')

    return layer(prev)
