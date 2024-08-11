#!/usr/bin/env python3
""" that creates a bag of words embedding matrix: """
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
        - sentences is a list of sentences to analyze
        - vocab is a list of the vocabulary words to use for the analysis
            - If None, all words within sentences should be used
        Returns: embeddings, features
        - embeddings numpy.ndarray of shape (s, f) containing the embeddings
            s is the number of sentences in sentences
            f is the number of features analyzed
        - features is a list of the features used for embeddings
        You are not allowed to use genism library.
    """

    def tokenize(sentence):
        # Remover contracciones y posesivos como 's
        sentence = re.sub(r"'s\b", '', sentence.lower())
        return re.findall(r'\b\w+\b', sentence)

    # Generar vocabulario si no está proporcionado
    if vocab is None:
        vocab = set()
        for sentence in sentences:
            vocab.update(tokenize(sentence))
        vocab = sorted(list(vocab))
    
    # Crear la matriz de embeddings con tipo de dato entero
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    # Llenar la matriz de embeddings
    for i, sentence in enumerate(sentences):
        words = tokenize(sentence)
        for word in words:
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1

    return embeddings, vocab
