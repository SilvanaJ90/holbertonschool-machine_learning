#!/usr/bin/env python3
""" that creates a pd.DataFrame from a np.ndarray: """
import pandas as pd
import numpy as np


def from_numpy(array):
    """
    - array is the np.ndarray from which you should create the pd.DataFrame
    - The columns of the pd.DataFrame should be labeled in alphabetical
        order and capitalized. There will not be more than 26 columns.
    Returns: the newly created pd.DataFrame
    """
    if not isinstance(array, np.ndarray):
        return ValueError('Input must be NumpyArray')

    num_columns = array.shape[1]
    if num_columns > 26:
        raise ValueError('The should not be more than  26 columns')

    num_labels = (chr(ord('A')+i) for i in range(num_columns))

    df = pd.DataFrame(array, columns=num_labels)

    return df
