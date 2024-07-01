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
    if vocab is None:
        vocab = set()
        for sentence in sentences:
            # Updated regex to avoid splitting possessive forms incorrectly
            words = re.findall(r"\b\w+(?:'\w+)?\b", sentence.lower())
            vocab.update(words)
        vocab = sorted(vocab)
    
    features = vocab
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)
    
    word_to_index = {word: i for i, word in enumerate(features)}
    
    for i, sentence in enumerate(sentences):
        # Updated regex to avoid splitting possessive forms incorrectly
        words = re.findall(r"\b\w+(?:'\w+)?\b", sentence.lower())
        for word in words:
            if word in word_to_index:
                embeddings[i][word_to_index[word]] += 1
    
    return embeddings, features