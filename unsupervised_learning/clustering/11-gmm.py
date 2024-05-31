#!/usr/bin/env python3
""" that calculates a GMM from a dataset """


import sklearn.mixture

def gmm(X, k):
    """ that calculates a GMM from a dataset """
    # Inicializar el modelo GMM
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)

    # Ajustar el modelo a los datos
    gmm_model.fit(X)

    # Extraer los parámetros del modelo GMM
    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariances_

    # Asignar los datos a los clusters
    clss = gmm_model.predict(X)

    # Calcular el BIC para diferentes tamaños de cluster
    kmin, kmax = 1, 10  # Cambia estos valores según sea necesario
    bic = []
    for k_test in range(kmin, kmax + 1):
        gmm_model_test = sklearn.mixture.GaussianMixture(n_components=k_test)
        gmm_model_test.fit(X)
        bic.append(gmm_model_test.bic(X))

    # Devolver los resultados esperados
    return pi, m, S, clss, bic
