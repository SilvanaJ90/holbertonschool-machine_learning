#!/usr/bin/env python3
""" that converts a gensim word2vec model to a keras Embedding layer """


def gensim_to_keras(model):
    """
        - model is a trained gensim word2vec models
        Returns: the trainable keras Embedding
    """
    return model.get_keras_embedding(train_embeddings=True)
