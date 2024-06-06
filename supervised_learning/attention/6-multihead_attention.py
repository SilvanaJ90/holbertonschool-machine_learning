#!/usr/bin/env python3
"""MultiHeadAttention implementation in TensorFlow."""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Class MultiHeadAttention to perform multi-head attention."""

    def __init__(self, dm, h):
        """
        Initialize the MultiHeadAttention layer.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        if dm % h != 0:
            raise ValueError("dm must be divisible by h")
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (h, depth).

        Args:
            x (tensor): Input tensor.
            batch_size (int): Batch size.

        Returns:
            Tensor with shape (batch_size, h, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask=None):
        """
        Perform the forward pass for multi-head attention.

        Args:
            Q (tensor): Query matrix of shape (batch, seq_len_q, dk).
            K (tensor): Key matrix of shape (batch, seq_len_v, dk).
            V (tensor): Value matrix of shape (batch, seq_len_v, dv).
            mask (tensor, optional): Mask to apply for attention.

        Returns:
            output (tensor): Attention output.
            attention_weights (tensor): Attention weights.
        """
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        scale_attention, attention_weights = sdp_attention(Q, K, V, mask)
        scale_attention = tf.transpose(
            scale_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scale_attention, (batch_size, -1, self.dm))

        output = self.linear(concat_attention)

        return output, attention_weights
