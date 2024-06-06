#!/usr/bin/env python3
""" Doc """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """ Class decoder """
    def __init__(
            self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
        """ DOc """
        super(Decoder, self).__init__()

        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(
            dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """ Doc """
        seq_len = tf.shape(x)[1]

        # Add embedding and positional encoding
        x = self.embedding(x)  # (batch_size, target_seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]

        x = self.dropout(x, training=training)

        # Pass through each decoder block
        for i in range(self.N):
            x = self.blocks[i](
                x, encoder_output, training, look_ahead_mask, padding_mask)

        return x
