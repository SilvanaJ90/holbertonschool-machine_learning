#!/usr/bin/env python3
""" calculate accurancy """
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """

    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction
    hint: accuracy = correct_predictions / all_predictions

    """

    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy
