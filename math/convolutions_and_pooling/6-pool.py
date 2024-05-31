#!/usr/bin/env python3
"""    that performs pooling on images """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Arg:
    - images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    - kernel_shape is a tuple of (kh, kw) containing
    the kernel shape for the pooling
        kh is the height of the kernel
        kw is the width of the kernel
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    mode indicates the type of pooling
        max indicates max pooling
        avg indicates average pooling
    You are only allowed to use two for loops; any other
    loops of any kind are not allowed
    Returns: a numpy.ndarray containing the pooled images



    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    pooled_images = np.zeros((m, output_h, output_w, c))

    for x in range(output_h):
        for y in range(output_w):
            x_start = x * sh
            x_end = x_start + kh
            y_start = y * sw
            y_end = y_start + kw

            if mode == 'max':
                pooled_images[:, x, y, :] = np.max(
                    images[:, x_start:x_end, y_start:y_end, :], axis=(1, 2))
            elif mode == 'avg':
                pooled_images[:, x, y, :] = np.mean(
                    images[:, x_start:x_end, y_start:y_end, :], axis=(1, 2))

    return pooled_images
