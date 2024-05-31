#!/usr/bin/env python3
""" Represents a cell of a simple RNN """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """  performs forward propagation for a simple RNN:

        rnn_cell is an instance of RNNCell that will
        be used for the forward propagation
        X is the data to be used, given as a
        numpy.ndarray of shape (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0 is the initial hidden state, given
        as a numpy.ndarray of shape (m, h)
            h is the dimensionality of the hidden state
        Returns: H, Y
            H is a numpy.ndarray containing all of the hidden states
            Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    H = [h_0]
    # Initialize the outputs array Y
    Y = []

    # Perform forward propagation through the RNN
    for step in X:
        h, y = rnn_cell.forward(H[-1], step)
        H.append(h)
        Y.append(y)

    return np.array(H), np.array(Y)
