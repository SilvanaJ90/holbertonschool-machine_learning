#!/usr/bin/env python3
""" Calculates the positional encoding for a transformer:"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    - max_seq_len is an integer representing the maximum sequence length
        dm is the model depth
    - Returns: a numpy.ndarray of shape (max_seq_len, dm)
        containing the positional encoding vectors
    """
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dimensions = np.arange(dm)[np.newaxis, :]
    angles = positions / (1000 ** (2 * (dimensions // 2) / np.float32(dm)))
    angles[:, ::2] = np.sin(angles[:, ::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return angles
