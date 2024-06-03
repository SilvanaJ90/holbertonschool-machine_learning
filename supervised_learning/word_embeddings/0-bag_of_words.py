#!/usr/bin/env python3
""" that creates a bag of words embedding matrix: """
import numpy as np
from collections import Counter


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
    tokenized_sentences = [sentence.lower().split() for sentence in sentences]

    # Get vocabulary if not provided
    if vocab is None:
        all_words = [word for sentence in tokenized_sentences for word in sentence]
        vocab = list(set(all_words))

    # Initialize embeddings matrix
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    # Fill the embeddings matrix
    for i, sentence in enumerate(tokenized_sentences):
        word_counts = Counter(sentence)
        for j, word in enumerate(vocab):
            embeddings[i, j] = word_counts[word]

    return embeddings, vocab
