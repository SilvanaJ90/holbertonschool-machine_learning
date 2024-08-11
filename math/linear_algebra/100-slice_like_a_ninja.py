#!/usr/bin/env python3


def np_slice(matrix, axes={}):
    """ that slices a matrix along specific axes: """
    slices = [slice(None)] * matrix.ndim

    # Update slices based on the axes dictionary
    for axis, slice_info in axes.items():
        slices[axis] = slice(*slice_info)

    # Convert list of slices to a tuple and apply to the matrix
    return matrix[tuple(slices)]
