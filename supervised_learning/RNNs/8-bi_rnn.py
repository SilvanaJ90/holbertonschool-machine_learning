#!/usr/bin/env python3
""" Doc """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.

    Parameters:
    - bi_cell: an instance of BidirectionalCell used for forward propagation
    - X: numpy.ndarray of shape (t, m, i) containing the data
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    - h_0: numpy.ndarray of shape (m, h) containing the
    initial hidden state in the forward direction
    - h_t: numpy.ndarray of shape (m, h) containing the
    initial hidden state in the backward direction

    Returns:
    - H: numpy.ndarray containing all the concatenated hidden states
    - Y: numpy.ndarray containing all the outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    # Initialize the forward and backward hidden states
    H_f = np.zeros((t, m, h))
    H_b = np.zeros((t, m, h))

    # Forward direction
    h_f = h_0
    for step in range(t):
        h_f = bi_cell.forward(h_f, X[step])
        H_f[step] = h_f

    # Backward direction
    h_b = h_t
    for step in range(t-1, -1, -1):
        h_b = bi_cell.backward(h_b, X[step])
        H_b[step] = h_b

    # Concatenate the hidden states from both directions
    H = np.concatenate((H_f, H_b), axis=-1)

    # Get the outputs
    Y = bi_cell.output(H)

    return H, Y
