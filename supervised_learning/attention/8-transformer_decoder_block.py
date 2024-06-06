#!/usr/bin/env python3
""" Doc """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """ Class DecoderBlock"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialize the DecoderBlock layer.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the fully connected layer.
            drop_rate (float): Dropout rate.
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Perform the forward pass for the decoder block.

        Args:
            x (tensor): Input tensor of shape (batch, target_seq_len, dm).
            encoder_output (tensor): Output tensor of shape
                (batch, input_seq_len, dm) from the encoder.
            training (bool): Determine if the model is training.
            look_ahead_mask (tensor, optional): Mask
                for the first multi-head attention layer.
            padding_mask (tensor, optional): Mask for the
                second multi-head attention layer.

        Returns:
            tensor: Output tensor of shape (batch, target_seq_len, dm).
        """
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        dense_output = self.dense_hidden(out2)
        dense_output = self.dense_output(dense_output)
        dense_output = self.dropout3(dense_output, training=training)
        out3 = self.layernorm3(out2 + dense_output)

        return out3
