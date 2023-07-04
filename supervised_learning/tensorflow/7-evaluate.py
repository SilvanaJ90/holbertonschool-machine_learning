#!/usr/bin/env python3

def evaluate(X, Y, save_path):
    """

    X is a numpy.ndarray containing the input data to evaluate
    Y is a numpy.ndarray containing the one-hot labels for X
    save_path is the location to load the model from
    Import the meta graph
    Get the following from the graph’s collection:
        tensors y_pred, loss, and accuracy
    You are not allowed to use tf.saved_model
    Returns: the network’s prediction, accuracy, and loss, respectively

    """