#!/usr/bin/env python3
""" DOc """
import numpy as np


class BidirectionalCell:
    """ Clas BidirectionalCell:"""
    def __init__(self, i, h, o):
        """
        Initialize the bidirectional cell

        Parameters:
        i - Dimensionality of the data
        h - Dimensionality of the hidden states
        o - Dimensionality of the outputs
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculate the hidden state in the forward
        direction for one time step

        Parameters:
        h_prev - numpy.ndarray of shape (m, h)
        containing the previous hidden state
        x_t - numpy.ndarray of shape (m, i)
        containing the data input for the cell

        Returns:
        h_next - The next hidden state
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(h_x, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        x_t is a numpy.ndarray of shape (m, i)
        that contains the data input for the cell
            m is the batch size for the data
        h_next is a numpy.ndarray of shape (m, h)
        containing the next hidden state
        Returns: h_pev, the previous hidden state
        """
        h_x = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(h_x, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """
        H is a numpy.ndarray of shape (t, m, 2 * h) that contains the
        concatenated hidden states from both directions,
        excluding their initialized states
            t is the number of time steps
            m is the batch size for the data
            h is the dimensionality of the hidden states
        Returns: Y, the outputs
        """
        t, m, _ = H.shape
        o = self.Wy.shape[1]
        Y = np.zeros((t, m, o))

        for step in range(t):
            Y[step] = self.softmax(np.matmul(H[step], self.Wy) + self.by)

        return Y

    def softmax(self, x):
        """Apply softmax activation function."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
