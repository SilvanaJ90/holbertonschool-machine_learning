#!/usr/bin/env python3
""" Doc """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Arguments:
    rnn_cells -- list of RNNCell instances of length l.
    X -- data to be used, numpy.ndarray of shape (t, m, i).
    h_0 -- initial hidden state, numpy.ndarray of shape (l, m, h).

    Returns:
    H -- numpy.ndarray containing all the hidden states.
    Y -- numpy.ndarray containing all the outputs.
    """
    t, m, i = X.shape
    l, _, h = h_0.shape

    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    for time_step in range(t):
        x_t = X[time_step]
        for layer in range(l):
            h_prev = H[time_step, layer]
            if layer == 0:
                H[time_step + 1, layer], _ = rnn_cells[
                    layer].forward(h_prev, x_t)
            else:
                H[time_step + 1, layer], _ = rnn_cells[
                    layer].forward(h_prev, H[time_step + 1, layer - 1])

    Y = []
    for time_step in range(t):
        _, y_t = rnn_cells[-1].forward(H[time_step + 1, -1], np.zeros((m, h)))
        Y.append(y_t)

    Y = np.stack(Y, axis=0)

    return H, Y
