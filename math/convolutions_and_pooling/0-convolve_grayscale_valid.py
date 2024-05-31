#!/usr/bin/env python3
"""  performs a valid convolution on grayscale images """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """

    - images is a numpy.ndarray with shape (m, h, w)
      containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    - kernel is a numpy.ndarray with shape (kh, kw)
      containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    You are only allowed to use two for loops;
    any other loops of any kind are not allowed
    Returns: a numpy.ndarray containing the convolved images

    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    # output_size=input_size−kernel_size+1
    output_height = h - kh + 1
    output_width = w - kw + 1
    # save images
    convolved_image = np.zeros((m, output_height, output_width))

    for x in range(output_height):
        for y in range(output_width):
            image = images[:, x:x+kh, y:y+kw]
            convolved_image[:, x, y] = np.sum(
                image * kernel, axis=(1, 2))
    return convolved_image
