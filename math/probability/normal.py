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
            self.stddev = (sum((x - self.mean)**2
                               for x in data) / len(data))**(1/2)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        x is the x-value
        Returns the z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        z is the z-score
        Returns the x-value of z
        """
        return z * self.stddev + self.mean

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
        return coef * e ** exp
    
    def cdf(self, x):
        """
        Calcula la función de distribución acumulativa (CDF)
        para un valor dado x utilizando una aproximación matemática.
        x es el valor para el cual se desea calcular la CDF.
        Devuelve la CDF para x.
        """
        t = 1 / (1 + 0.3275911 * (x - self.mean) / self.stddev)
        cdf_value = 1 - 0.3989423 * self.exp_approx(-((x - self.mean) / self.stddev) ** 2 / 2) * (
                t * (0.254829592 - t * (-0.284496736 - t * (1.421413741 - t * (-1.453152027 - t * 1.061405429)))))

        return cdf_value

    def exp_approx(self, x):
        """
        Aproximación de la función exponencial utilizando una serie de Taylor truncada.
        x es el valor para el cual se desea calcular la función exponencial.
        Devuelve la aproximación de la función exponencial para x.
        """
        # Truncar la serie de Taylor después del término cuadrático
        return 1 + x + (x ** 2) / 2