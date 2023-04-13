#!/usr/bin/env python3
""" Doc """


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial:
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    result = []
    for i in range(1, len(poly)):
        coeff = i * poly[i]
        result.append(coeff)

    if sum(result) == 0:
        return [0]
    return result
