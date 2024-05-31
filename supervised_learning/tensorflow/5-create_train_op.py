#!/usr/bin/env python3
""" Create train op """
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """

    loss is the loss of the network’s prediction
    alpha is the learning rate
    Returns: an operation that trains the network using gradient descent

    """
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(
        learning_rate=alpha, name='GradientDescent')

    return optimizer.minimize(loss)
