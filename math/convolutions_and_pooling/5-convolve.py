#!/usr/bin/env python3
"""   performs a convolution on images using multiple kernels """
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Arg:

    - images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    - kernels is a numpy.ndarray with shape (kh, kw, c, nc)
    containing the kernels for the convolution
        kh is the height of a kernel
        kw is the width of a kernel
        nc is the number of kernels
    padding is either a tuple of (ph, pw), 'same', or 'valid'
        if 'same', performs a same convolution
        if 'valid', performs a valid convolution
        if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
        the image should be padded with 0's
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    You are only allowed to use three for loops;
    any other loops of any kind are not allowed
    Returns: a numpy.ndarray containing the convolved images


    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    else:
        ph, pw = padding

    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, output_h, output_w, nc))

    images_padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    for x in range(output_h):
        for y in range(output_w):
            for k in range(nc):
                sliced_images = images_padded[:, x * sh:x * sh + kh,
                                              y * sw:y * sw + kw, :]
                output[:, x, y, k] = np.sum(
                    kernels[:, :, :, k] * sliced_images, axis=(1, 2, 3))

    return output
