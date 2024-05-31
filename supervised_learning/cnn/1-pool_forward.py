#!/usr/bin/env python3
"""  that performs forward propagation over a pooling layer of a neural network
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Realiza la propagación hacia adelante sobre
    una capa de agrupamiento de una red neuronal.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calcula las dimensiones de la salida
    h_output = (h_prev - kh) // sh + 1
    w_output = (w_prev - kw) // sw + 1

    # Inicializa la matriz de salida
    A = np.zeros((m, h_output, w_output, c_prev))

    # Itera sobre cada ejemplo de entrada
    for i in range(m):
        # Itera sobre cada canal de entrada
        for c in range(c_prev):
            # Itera sobre cada posición en la salida
            for h in range(h_output):
                for w in range(w_output):
                    # Selecciona la ventana de entrada
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw
                    a_prev_slice = A_prev[i, vert_start:vert_end,
                                          horiz_start:horiz_end, c]

                    # Aplica la operación de agrupamiento (max o avg)
                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'avg':
                        A[i, h, w, c] = np.mean(a_prev_slice)

    return A
