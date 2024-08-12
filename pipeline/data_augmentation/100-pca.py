#!/usr/bin/env python3
""" that performs PCA color augmentation as
    described in the AlexNet paper:
"""
import tensorflow as tf


def pca_color(image, alphas):
    """
    image is a 3D tf.Tensor containing the image to change
    alphas a tuple of length 3 containing the amount
        that each channel should change
    Returns the augmented image
    """
