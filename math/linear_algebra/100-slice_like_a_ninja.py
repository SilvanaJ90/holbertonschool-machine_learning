#!/usr/bin/env python3
""" that slices a matrix along specific axes: """


def np_slice(matrix, axes={}):
    """
    You can assume that matrix is a numpy.ndarray
    You must return a new numpy.ndarray
    axes is a dictionary where the key is an axis
    to slice along and the value is a tuple representing the slice to make
    along that axis You can assume that axes represents a valid slice
    Hint
    """
    slices = [slice(None)] * matrix.ndim

    # Update slices based on the axes dictionary
    for axis, slice_info in axes.items():
        slices[axis] = slice(*slice_info)

    # Convert list of slices to a tuple and apply to the matrix
    return matrix[tuple(slices)]
