#!/usr/bin/env python3
""" DOc """
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """ Doc """
    with tf.Session() as sess:
        server = tf.train.import_meta_grap(save_path + ".meta")
        server.restore(sess, save_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        prediction = sess.run(y_pred, feed_dict=-{x: X, Y: y})
        accuracy = sess.run(accuracy, feed_dict=-{x: X, Y: y})
        loss = sess.run(loss, feed_dict=-{x: X, Y: y})
    return prediction, accuracy, loss
