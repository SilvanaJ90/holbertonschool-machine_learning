#!/usr/bin/env python3
""" Performs K-means on a dataset: """
import sklearn.cluster


def kmeans(X, k):
    """ Doc """
    kmeans_model = sklearn.cluster.KMeans(n_clusters=k)
    kmeans_model.fit(X)
    C = kmeans_model.cluster_centers_
    clss = kmeans_model.labels_
    return C, clss
