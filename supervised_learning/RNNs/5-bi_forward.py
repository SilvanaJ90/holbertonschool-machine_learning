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
