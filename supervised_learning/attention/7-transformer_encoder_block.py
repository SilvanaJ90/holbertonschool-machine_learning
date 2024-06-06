#!/usr/bin/env python3
"""  create an encoder block for a transformer: """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """ Class ENcoderBlock """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialize the EncoderBlock layer.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the fully connected layer.
            drop_rate (float): Dropout rate.
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Perform the forward pass for the encoder block.

        Args:
            x (tensor): Input tensor of shape (batch, input_seq_len, dm).
            training (bool): Determine if the model is training.
            mask (tensor, optional):
                Mask to be applied for multi-head attention.

        Returns:
            tensor: Output tensor of shape (batch, input_seq_len, dm).
        """
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        dense_output = self.dense_hidden(out1)
        dense_output = self.dense_output(dense_output)
        dense_output = self.dropout2(dense_output, training=training)
        out2 = self.layernorm2(out1 + dense_output)

        return out2
