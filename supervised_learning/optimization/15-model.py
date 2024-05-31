#!/usr/bin/env python3
"""
That builds, trains, and saves a neural network model in tensorflow
using Adam optimization, mini-batch gradient descent,
learning rate decay, and batch normalization:
"""
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


def shuffle_data(X, Y):
    """
    fill the function
    """
    m = X.shape[0]
    shuffle_vector = list(np.random.permutation(m))
    x = X[shuffle_vector, :]
    y = Y[shuffle_vector, :]
    return x, y


def create_layer(prev, n, activation):
    """
    prev is the tensor output of the previous layer
    Returns: the tensor output of the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.layers.Dense(
        units=n, kernel_initializer=init, name='layer')

    return layer(prev)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Doc
    """
    learning = tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True)
    return learning


def create_Adam_op(loss, alpha, beta1, beta2, epsilon, global_step):
    """
    Arg:
    """
    op_a = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return op_a.minimize(loss, global_step=global_step)


def create_batch_norm_layer(prev, n, activation):
    """
    Doc
    """
    activa = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, activation=None,
                                  kernel_initializer=activa, name='layer')
    Z = layer(prev)
    mu, sigma_2 = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(initial_value=tf.constant(
        1.0, shape=[n]), name='gamma')
    beta = tf.Variable(initial_value=tf.constant(
        0.0, shape=[n]), name='beta')
    z_b_norm = tf.nn.batch_normalization(
        Z, mu,
        sigma_2,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-8)
    return activation(z_b_norm)


def forward_prop(prev, layers, activations):
    """
    all layers get batch_normalization but the last one,
    that stays without any activation or normalization
    """
    for i, n in enumerate(layers[: -1]):
        prev = create_batch_norm_layer(prev, n, activations[i])
    prev = create_layer(prev, layers[-1], activations[-1])
    return prev


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


def calculate_loss(y, y_pred):
    """
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the networkis predictions
    Returns: a tensor containing the loss of the prediction

    """

    return tf.compat.v1.losses.softmax_cross_entropy(y, y_pred)


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """ Doc """
    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    m, nx = X_train.shape
    classes = Y_train.shape[1]

    # initialize x, y and add them to collection
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    # initialize y_pred and add it to collection
    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    # intialize loss and add it to collection
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    # intialize accuracy and add it to collection
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    # intialize global_step variable
    # hint: not trainable
    global_step = tf.Variable(0, trainable=False)

    # compute decay_steps
    decay_steps = m // batch_size
    if m % batch_size:
        decay_steps += 1

    # create "alpha" the learning rate decay operation in tensorflow
    alpha = learning_rate_decay(alpha, decay_rate, global_step, decay_steps)

    # initizalize train_op and add it to collection
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon, global_step)
    tf.add_to_collection('train_op', train_op)

    # hint: don't forget to add global_step parameter in optimizer().minimize()

    init = tf.global_variables_initializer()
    with tf.compat.v1.Session() as sess:

        sess.run(init)

        for i in range(epochs):
            # print training and validation cost and accuracy
            train_cost, train_accuracy = sess.run(
                (loss, accuracy), feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run(
                (loss, accuracy), feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            # shuffle data
            X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)

            for j in range(0, X_train.shape[0], batch_size):
                # get X_batch and Y_batch
                X_batch = X_shuffle[j:j + batch_size]
                Y_batch = Y_shuffle[j:j + batch_size]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                # run training operation
                if not ((j // batch_size + 1) % 100):
                    cost, acc = sess.run(
                        (loss, accuracy), feed_dict={x: X_batch, y: Y_batch})

                    # print batch cost and accuracy
                    print("\tStep {}:".format(j // batch_size + 1))
                    print("\t\tCost: {}".format(cost))
                    print("\t\tAccuracy: {}".format(acc))

        # print training and validation cost and accuracy again
        train_cost, train_accuracy = sess.run(
                (loss, accuracy), feed_dict={x: X_train, y: Y_train})
        valid_cost, valid_accuracy = sess.run(
                (loss, accuracy), feed_dict={x: X_valid, y: Y_valid})

        print("After {} epochs:".format(epochs))
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_accuracy))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_accuracy))

        # save and return the path to where the model was saved
        saver = tf.train.Saver()
        return saver.save(sess, save_path)
