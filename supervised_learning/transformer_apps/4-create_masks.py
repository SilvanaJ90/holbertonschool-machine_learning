#!/usr/bin/env python3
""" This script creates all masks for training/validation """
import tensorflow.compat.v2 as tf

def create_padding_mask(seq):
    """
    Create a padding mask for a given sequence.
    Args:
    - seq: a tf.Tensor of shape (batch_size, seq_len) containing the input sequence
    Returns:
    - A tf.Tensor of shape (batch_size, 1, 1, seq_len) containing the padding mask
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    """
    Create a look-ahead mask for a sequence of a given size.
    Args:
    - size: an integer representing the length of the sequence
    Returns:
    - A tf.Tensor of shape (size, size) containing the look-ahead mask
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inputs, target):
    """
    Create all masks for training/validation.
    Args:
    - inputs: a tf.Tensor of shape (batch_size, seq_len_in) containing the input sentence
    - target: a tf.Tensor of shape (batch_size, seq_len_out) containing the target sentence
    Returns:
    - enc_padding_mask: a tf.Tensor padding mask of shape (batch_size, 1, 1, seq_len_in) for the encoder
    - combined_mask: a tf.Tensor of shape (batch_size, 1, seq_len_out, seq_len_out) for the first attention block in the decoder
    - dec_padding_mask: a tf.Tensor padding mask of shape (batch_size, 1, 1, seq_len_in) for the second attention block in the decoder
    """
    enc_padding_mask = create_padding_mask(inputs)

    dec_padding_mask = create_padding_mask(inputs)

    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    dec_target_padding_mask = create_padding_mask(target)

    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask
