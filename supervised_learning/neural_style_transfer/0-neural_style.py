#!/usr/bin/env python3

import numpy as np
import tensorflow as tf


class NST:
    """ Class nst """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """ Constructor """

        if not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = NST.scale_image(style_image)
        self.content_image = NST.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        
    @staticmethod
    def scale_image(image):
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        h, w = image.shape[:2]
        max_dim = 512
        max_side = max(h, w)
        scale = max_dim / max_side

        new_h = round(h * scale)
        new_w = round(w * scale)

        image = tf.image.resize(image, (new_h, new_w), method=tf.image.ResizeMethod.BICUBIC)
        image /= 255.0  # Rescaling pixel values to [0, 1]
        image = tf.expand_dims(image, axis=0)  # Adding batch dimension
        return image
