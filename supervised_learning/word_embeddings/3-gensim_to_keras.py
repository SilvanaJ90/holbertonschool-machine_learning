#!/usr/bin/env python3
""" that converts a gensim word2vec model to a keras Embedding layer """
from keras.layers import Embedding
import numpy as np


def gensim_to_keras(model):
    """
        - model is a trained gensim word2vec models
        Returns: the trainable keras Embedding
    """
    weights = model.wv.vectors

    # Get the size of the vocabulary and the size of the embeddings
    vocab_size, embedding_size = weights.shape

    # Create the Keras Embedding layer
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        weights=[weights]
        )

    return embedding_layer
