#!/usr/bin/env python3
""" Doc """


class Normal:
    """ Class Normal """
    def __init__(self, data=None, mean=0., stddev=1.):
        self.mean = float(mean)
        self.stddev = float(stddev)

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            variation = (sum((x - self.mean)**2 for x in data) / len(data))
            self.stddev = (variation**0.5)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        x is the x-value
        Returns the z-score of x
        """
        z_score = (x - self.mean) / self.stddev
        return z_score

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        z is the z-score
        Returns the x-value of z
        """
        x_value = z * self.stddev + self.mean
        return x_value

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        x is the x-value
        Returns the PDF value for x
        """
        e = 2.7182818285
        π = 3.1415926536
        coef = 1 / (self.stddev * (2 * π) ** 0.5)
        exp = -(x - self.mean) ** 2 / (2 * self.stddev ** 2)
        pdf_value = coef * e ** exp
        return pdf_value

    def erf(self, x):
        """
        Calculates the value of the error function for a given x-value
        x is the x-value
        Returns the error function value for x
        """
        π = 3.1415926536
        erf_value = 2/π**0.5*(x-(x**3/3)+(x**5/10)-(x**7/42)+(x**9/216))
        return erf_value

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        x is the x-value
        Returns the CDF value for x
        """
        sqrt2 = 2**0.5
        arg = (x - self.mean) / (self.stddev * sqrt2)
        erf_value = self.erf(arg)
        cdf_value = 0.5 * (1 + erf_value)
        return cdf_value
