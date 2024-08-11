#!/usr/bin/env python3
""" that adds two matrices: """


def add_matrices(mat1, mat2):
    """

    You can assume that mat1 and mat2 are matrices containing ints/floats
    You can assume all elements in the same dimension type/shape
    You must return a new matrix
    If matrices are not the same shape, return None
    You can assume that mat1 and mat2 will never be empty

    """
    def check_shape(m1, m2):
        """ doC """
        if isinstance(m1, list) and isinstance(m2, list):
            if len(m1) != len(m2):
                return False
            return all(check_shape(sub1, sub2) for sub1, sub2 in zip(m1, m2))
        return True

    # Perform element-wise addition of matrices
    def add(m1, m2):
        if isinstance(m1, list) and isinstance(m2, list):
            return [add(sub1, sub2) for sub1, sub2 in zip(m1, m2)]
        return m1 + m2

    if not check_shape(mat1, mat2):
        return None

    return add(mat1, mat2)
