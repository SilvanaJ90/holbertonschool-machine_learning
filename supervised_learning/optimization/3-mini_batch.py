#!/usr/bin/env python3
"""
    Trains a loaded neural network model
    using mini-batch gradient descent
"""
import tensorflow.compat.v1 as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(
        X_train, Y_train, X_valid, Y_valid,
        batch_size=32, epochs=5,
        load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """
        Trains a loaded neural network model
        using mini-batch gradient descent
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path+'.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for i in range(epochs):
            print("After {} epochs:".format(i))
            train_cost, train_accuracy = sess.run(
                (loss, accuracy), feed_dict={x: X_train, y: Y_train})
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            valid_cost, valid_accuracy = sess.run(
                (loss, accuracy), feed_dict={x: X_valid, y: Y_valid})
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
            x_shuffle, y_shuffle = shuffle_data(X_train, Y_train)

            for j in range(0, X_train.shape[0], batch_size):
                x_batch = x_shuffle[j:j + batch_size]
                y_batch = y_shuffle[j:j + batch_size]
                sess.run(train_op, feed_dict={x: x_batch, y: y_batch})
                if not ((j // batch_size + 1) % 100):
                    cost, acc = sess.run(
                        (loss, accuracy), feed_dict={x: x_batch, y: y_batch})
                    print("\tStep {}:".format(j // batch_size + 1))
                    print("\t\tCost: {}".format(cost))
                    print("\t\tAccuracy: {}".format(acc))

        print("After {} epochs:".format(epochs))
        train_cost, train_accuracy = sess.run(
            (loss, accuracy), feed_dict={x: X_train, y: Y_train})
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_accuracy))
        valid_cost, valid_accuracy = sess.run(
            (loss, accuracy), feed_dict={x: X_valid, y: Y_valid})
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_accuracy))

        return saver.save(sess, save_path)
