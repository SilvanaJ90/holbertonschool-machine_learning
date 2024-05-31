#!/usr/bin/env python3
""" DOc """
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """ Doc """
    x, y = create_placeholders(X.shape[1], Y.shape[1])
    y_pred = forward_prop(
        x, [256, 256, Y.shape[1]], [tf.nn.tanh, tf.nn.tanh, None])
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        Y_pred, acc, cost = sess.run(
            [y_pred, accuracy, loss], feed_dict={x: X, y: Y})

    return Y_pred, acc, cost
