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
    # Preprocess sentences: convert to lowercase and split into words
    processed_sentences = [re.findall(r'\b\w+\b', sentence.lower()) for sentence in sentences]

    # If no vocabulary is provided, generate it from the sentences
    if vocab is None:
        vocab = sorted(set(word for sentence in processed_sentences for word in sentence))

    # Initialize the embeddings matrix with zeros
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    # Fill the embeddings matrix
    for i, sentence in enumerate(processed_sentences):
        for word in sentence:
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1

    return embeddings, vocab
