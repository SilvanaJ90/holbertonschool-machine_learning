#!/usr/bin/env python3
""" performs forward propagation over a convolutional
    layer of a neural network
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Perform forward propagation over a convolutional layer of a neural network.

    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Inicializar ph y pw con valores predeterminados
    ph, pw = 0, 0

    if padding == "same":
        # Calcular el tamaño de relleno para mantener el mismo tamaño de salida
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2

    # Calcular el tamaño de salida
    h_output = (h_prev + 2 * ph - kh) // sh + 1
    w_output = (w_prev + 2 * pw - kw) // sw + 1

    # Inicializar la matriz de salida con las dimensiones correctas
    Z = np.zeros((m, h_output, w_output, c_new))

    # Aplicar relleno a la matriz de entrada
    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    for x in range(h_output):
        for y in range(w_output):
            for k in range(c_new):
                # Seleccionar la ventana de entrada
                vert_start = x * sh
                vert_end = vert_start + kh
                horiz_start = y * sw
                horiz_end = horiz_start + kw
                a_slice_prev = A_prev_pad[:, vert_start:vert_end,
                                          horiz_start:horiz_end, :]

                # Convolución
                Z[:, x, y, k] = np.sum(a_slice_prev * W[:, :, :, k],
                                       axis=(1, 2, 3)) + b[:, :, :, k]

    # Aplicar la función de activación
    A = activation(Z)

    return A
