#!/usr/bin/env python3
""" That flips an image horizontally: """
import tensorflow as tf


def flip_image(image):
    """
    image is a 3D tf.Tensor containing the image to flip
    Returns the flipped image
    """
    return tf.image.flip_left_right(image)
