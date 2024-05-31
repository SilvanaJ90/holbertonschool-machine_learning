#!/usr/bin/env python3
""" that represents an LSTM unit """
import numpy as np


class LSTMCell:
    """ class STMCell"""
    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes
        Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by
        that represent the weights and biases of the cell
            Wfand bf are for the forget gate
            Wuand bu are for the update gate
            Wcand bc are for the intermediate cell state
            Woand bo are for the output gate
            Wyand by are for the outputs
        The weights should be initialized using a random
        normal distribution in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        # Bias vectors
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        x_t is a numpy.ndarray of shape (m, i) that
        contains the data input for the cell
            m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h)
        containing the previous hidden state
        c_prev is a numpy.ndarray of shape (m, h)
        containing the previous cell state
        The output of the cell should use a softmax activation function
        Returns: h_next, c_next, y
            h_next is the next hidden state
            c_next is the next cell state
            y is the output of the cell
        """
        # Concatenate x_t and h_prev to form [x_t, h_prev]
        concat = np.concatenate((x_t, h_prev), axis=1)

        # Forget gate
        ft = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)

        # Update gate
        ut = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)

        # Intermediate cell state
        cct = np.tanh(np.matmul(concat, self.Wc) + self.bc)

        # Cell state
        c_next = ft * c_prev + ut * cct

        # Output gate
        ot = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)

        # Hidden state
        h_next = ot * np.tanh(c_next)

        # Output
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y

    @staticmethod
    def sigmoid(x):
        """
        Compute the sigmoid of x.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """
        Compute the softmax of x.
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
