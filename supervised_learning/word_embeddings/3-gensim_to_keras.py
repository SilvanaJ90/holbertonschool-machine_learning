#!/usr/bin/env python3
""" that converts a gensim word2vec model to a keras Embedding layer """
import numpy as np


def gensim_to_keras(model):
    """
        - model is a trained gensim word2vec models
        Returns: the trainable keras Embedding
    """
