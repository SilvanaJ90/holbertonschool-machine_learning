#!/usr/bin/env python3
"""  performs a valid convolution on grayscale images """
import numpy as np


def convolve_grayscale_same(images, kernel):
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
    if necessary, the image should be padded with 0`s
    You are only allowed to use two for loops;
    any other loops of any kind are not allowed
    Returns: a numpy.ndarray containing the convolved images

    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    # Calculate padding size p=f−1​ / 2
    pad_height = int((kh - 1) / 2) if kh % 2 != 0 else int(kh / 2)
    pad_width = int((kw - 1) / 2) if kw % 2 != 0 else int(kw / 2)

    output = np.zeros(shape=(m, h, w))
    # Apply padding
    images_padded = np.pad(
        images,
        ((0, 0),
         (pad_height, pad_height),
         (pad_width, pad_width)),
        mode="constant"
        )

    for x in range(h):
        for y in range(w):
            output[:, x, y] = np.sum(
                (kernel * images_padded[:, x:x + kh, y:y + kw]),
                axis=(1, 2)
            )

    return output
