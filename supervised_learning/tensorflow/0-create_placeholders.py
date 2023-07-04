#!/usr/bin/env python3
import tensorflow as tf
""" Function create_placeholders """


def create_placeholders(nx, classes):
    """ Doc """
    x = tf.placeholder("float", (None, nx), name='x')
    y = tf.placeholder("float", (None, classes), name='y')
    return x, y
