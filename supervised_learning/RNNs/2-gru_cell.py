#!/usr/bin/env python3
""" class GRUCell """
import numpy as np


class GRUCell:
    """ represents a gated recurrent unit """

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
        that represent the weights and biases of the cell
            Wzand bz are for the update gate
            Wrand br are for the reset gate
            Whand bh are for the intermediate hidden state
            Wyand by are for the output
        The weights should be initialized using a random normal
        distribution in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros

        """

        # Initialize weights and biases for the update gate
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))

        # Initialize weights and biases for the reset gate
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))

        # Initialize weights and biases for the intermediate hidden state
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))

        # Initialize weights and biases for the output
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        x_t is a numpy.ndarray of shape (m, i)
        that contains the data input for the cell
            m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h)
        containing the previous hidden state
        The output of the cell should use a softmax activation function
        Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        # Concatenate the input and the previous hidden state
        h_x = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z_t = self.sigmoid(np.matmul(h_x, self.Wz) + self.bz)

        # Reset gate
        r_t = self.sigmoid(np.matmul(h_x, self.Wr) + self.br)

        # Intermediate hidden state
        h_x_r = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(h_x_r, self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # Output
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y

    def sigmoid(self, x):
        """ func sigmoid """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """ fuct act softmax"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
