#!/usr/bin/env python3
"""  to create the decoder for a transformer: """
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Embedding, Dropout
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Encoder(Layer):
    """ class Encoder """
    def __init__(
            self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        """ DOc """
        super(Encoder, self).__init__()

        self.N = N
        self.dm = dm
        self.embedding = Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(
            dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = Dropout(drop_rate)

    def call(self, x, training, mask):
        """ Doc """
        seq_len = tf.shape(x)[1]

        # Add embedding and positional encoding
        x = self.embedding(x)  # (batch_size, input_seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]

        x = self.dropout(x, training=training)

        # Pass through each encoder block
        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x
