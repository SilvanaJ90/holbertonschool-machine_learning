#!/usr/bin/env python3
""" Function create_placeholders """
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


def create_placeholders(nx, classes):
    """
        Fuction return placeholders x and y
        for the neural network
        nx: the number of feature columns in our data
        classes: the number of classes in our classifier
        Returns: placeholders named x and y, respectively
        x is the placeholder for the input data to the neural network
        y is the placeholder for the one-hot labels for the input data
    """

    x = tf.placeholder("float32", (None, nx), name='x')
    y = tf.placeholder("float32", (None, classes), name='y')
    
    return x, y
