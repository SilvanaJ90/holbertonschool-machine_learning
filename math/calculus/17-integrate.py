#!/usr/bin/env python3
""" Doc """


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial:
    """
    if not isinstance(poly, list) or \
       not all(isinstance(coeff, (int, float)) for coeff in poly) or \
       not isinstance(C, int):
        return None

    if poly == [] or C is None:
        return None

    integral_coeffs = [C]
    for i, coeff in enumerate(poly, start=1):
        integral = coeff / i
        if integral.is_integer():
            integral_coeffs.append(int(integral))
        else:
            integral_coeffs.append(integral)

    return integral_coeffs
