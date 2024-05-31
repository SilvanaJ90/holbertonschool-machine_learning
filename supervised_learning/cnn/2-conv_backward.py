#!/usr/bin/env python3
"""  that performs back propagation over a convolutional
 layer of a neural network:
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Realiza la retropropagación sobre
    una capa de convolución de una red neuronal.
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Initialize gradients
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    # Compute padding
    if padding == 'same':
        pad_h = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pad_w = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    elif padding == 'valid':
        pad_h, pad_w = (0, 0)
    else:
        pad_h, pad_w = padding

    # Pad A_prev and dA_prev
    A_prev_pad = np.pad(
        A_prev,
        (
         (0, 0),
         (pad_h,
          pad_h),
         (pad_w, pad_w),
         (0, 0)), mode='constant'
          )
    dA_prev_pad = np.pad(
        dA_prev, ((0, 0), (pad_h, pad_h),
                  (pad_w, pad_w), (0, 0)), mode='constant')

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = a_prev_pad[
                        vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients
                    da_prev_pad[vert_start:vert_end,
                                horiz_start:horiz_end,
                                :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        # Assign gradient of the unpadded region to dA_prev
        if padding == 'valid':
            dA_prev[i] = da_prev_pad
        else:
            dA_prev[i, :, :, :] = da_prev_pad[pad_h:-pad_h, pad_w:-pad_w, :]

    return dA_prev, dW, db
