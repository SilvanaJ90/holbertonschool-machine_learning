#!/usr/bin/env python3
""" Train neural nework"""
import tensorflow.compat.v1 as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Doc

    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    # Add tensors to collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    # Initialize variables and create a Saver
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            if i > 0:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
            
            # Compute training and validation metrics
            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            # Print metrics
            if i % 100 == 0 or i == 0 or i == iterations:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_accuracy}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_accuracy}")

        # Save the trained model
        save_path = saver.save(sess, save_path)

    return save_path
