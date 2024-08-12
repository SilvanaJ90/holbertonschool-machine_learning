#!/usr/bin/env python3
"""  that rotates an image by 90 degrees counter-clockwise: """
import tensorflow as tf


def rotate_image(image):
    """   
    image is a 3D tf.Tensor containing the image to rotate
    Returns the rotated image
    """
    return tf.image.rot90(image, k=1, name=None)
