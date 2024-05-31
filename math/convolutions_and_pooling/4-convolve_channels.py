#!/usr/bin/env python3
"""   that performs a convolution on images with channels: """
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Arg:
    - images is a numpy.ndarray with shape (m, h, w)
      containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    - kernel is a numpy.ndarray with shape (kh, kw)
      containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    - padding is either a tuple of (ph, pw), `same`, or 'valid'
        if 'same', performs a same convolution
        if 'valid', performs a valid convolution
        if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
        the image should be padded with 0's
    - stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    You are only allowed to use two for loops; any other
    loops of any kind are not allowed
    Returns: a numpy.ndarray containing the convolved images

    """
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
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

    output = np.zeros((m, output_h, output_w))

    images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    for x in range(output_h):
        for y in range(output_w):
            output[:, x, y] = np.sum(
                kernel * images[:, x * sh:x * sh + kh, y * sw:y * sw + kw, :],
                axis=(1, 2, 3))

    return output
