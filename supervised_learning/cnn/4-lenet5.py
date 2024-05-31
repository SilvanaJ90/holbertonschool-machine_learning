#!/usr/bin/env python3
"""  hat builds a modified version of the
LeNet-5 architecture using tensorflow::
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """ doc """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Convolutional layer 1
    conv1 = tf.layers.Conv2D(
        filters=6, kernel_size=(5, 5), padding='same',
        activation=tf.nn.relu, kernel_initializer=initializer)(x)

    # Max pooling layer 1
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # Convolutional layer 2
    conv2 = tf.layers.Conv2D(
        filters=16, kernel_size=(5, 5), padding='valid',
        activation=tf.nn.relu, kernel_initializer=initializer)(pool1)

    # Max pooling layer 2
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Flatten
    flatten = tf.layers.Flatten()(pool2)

    # Fully connected layer 1
    fc1 = tf.layers.Dense(
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=initializer)(flatten)

    # Fully connected layer 2
    fc2 = tf.layers.Dense(
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=initializer)(fc1)

    # Output layer
    logits = tf.layers.Dense(units=10, kernel_initializer=initializer)(fc2)

    # Softmax activation
    softmax = tf.nn.softmax(logits)

    # Loss function
    loss = tf.losses.softmax_cross_entropy(y, logits)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Training operation
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return softmax, train_op, loss, accuracy
