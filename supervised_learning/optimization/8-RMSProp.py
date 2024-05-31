#!/usr/bin/env python3
"""
    creates the training operation
    for a neural network in tensorflow using the RMSProp
"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Arg:

        loss is the loss of the network
        alpha is the learning rate
        beta2 is the RMSProp weight
        epsilon is a small number to avoid division by zero
        Returns: the RMSProp optimization operation

    """
    op = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    return op.minimize(loss)
