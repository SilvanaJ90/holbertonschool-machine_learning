#!/usr/bin/env python3

def create_train_op(loss, alpha):
    """

    loss is the loss of the network’s prediction
    alpha is the learning rate
    Returns: an operation that trains the network using gradient descent

    """