#!/usr/bin/env python3
"""  to create a transformer network: """
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """ Class transformer"""
    def __init__(
            self, N, dm, h, hidden, input_vocab, target_vocab,
            max_seq_input, max_seq_target, drop_rate=0.1):
        """ Doc """
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(
            self, inputs, target,
            training, encoder_mask,
            look_ahead_mask, decoder_mask):
        """ doc """
        # Encode the inputs
        encoder_output = self.encoder(inputs, training, encoder_mask)

        # Decode the encoded inputs
        decoder_output = self.decoder(target, encoder_output,
                                      training, look_ahead_mask, decoder_mask)

        # Final linear layer to get the predictions
        final_output = self.linear(decoder_output)

        return final_output
