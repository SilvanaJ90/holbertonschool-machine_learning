#!/usr/bin/env python3
""" Represents a cell of a simple RNN """
import numpy as np


class RNNCell:
    """ class RNNCell """
    def __init__(self, i, h, o):
        """
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs
            Creates the public instance attributes
            Wh, Wy, bh, by that represent the weights and biases of the cell
            Wh and bh are for the concatenated hidden state and input data
            Wy and by are for the output

        """
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ That performs forward propagation for one time step
            x_t is a numpy.ndarray of shape (m, i) that
            contains the data input for the cell
                m is the batche size for the data
            h_prev is a numpy.ndarray of shape (m, h)
            containing the previous hidden state
            The output of the cell should use a softmax activation function
            Returns: h_next, y
                h_next is the next hidden state
                y is the output of the cell
        """
        # Concatenate h_prev and x_t hprev,xt]
        h_x_concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute the next hidden state hnext​=tanh(Wh⋅[hprev​,xt​]+bh)
        h_next = np.tanh(np.dot(h_x_concat, self.Wh) + self.bh)

        # Compute the cell output y=softmax(Wy⋅hnext​+by).
        y_linear = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)
        return h_next, y

    @staticmethod
    def softmax(x):
        """
        Computes softmax activation function

        Parameters:
        x (numpy.ndarray): Linear outputs

        Returns:
        numpy.ndarray: Softmax outputs
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
