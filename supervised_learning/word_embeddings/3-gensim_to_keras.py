#!/usr/bin/env python3
""" that converts a gensim word2vec model to a keras Embedding layer """
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences


def gensim_to_keras(model):
    """
        - model is a trained gensim word2vec models
        Returns: the trainable keras Embedding
    """
