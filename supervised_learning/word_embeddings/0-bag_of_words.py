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
        sentence = sentence.lower()
        words = re.findall(r'\b\w+\b', sentence)
        return words

    # Tokenize all sentences
    tokenized_sentences = [tokenize(sentence) for sentence in sentences]

    # If vocab is None, build it from the sentences
    if vocab is None:
        vocab_set = set()
        for sentence in tokenized_sentences:
            vocab_set.update(sentence)
        vocab = sorted(vocab_set)

    # Create a word index dictionary for quick lookup
    word_index = {word: idx for idx, word in enumerate(vocab)}

    # Initialize the embeddings matrix with zeros
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    # Populate the embeddings matrix
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in word_index:
                embeddings[i, word_index[word]] += 1

    return embeddings, vocab
