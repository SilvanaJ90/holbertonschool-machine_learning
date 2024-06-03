#!/usr/bin/env python3
""" that creates a bag of words embedding matrix: """
import numpy as np


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
    word_counts = {}
    tokenized_sentences = []
    for sentence in sentences:
        tokens = sentence.lower().split()
        tokenized_sentences.append(tokens)
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1

    # Create vocabulary if not provided
    if vocab is None:
        vocab = sorted(word_counts.keys())

    # Initialize embeddings matrix
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    # Fill embeddings matrix
    for i, tokens in enumerate(tokenized_sentences):
        for j, word in enumerate(vocab):
            embeddings[i, j] = tokens.count(word)

    return embeddings, vocab
