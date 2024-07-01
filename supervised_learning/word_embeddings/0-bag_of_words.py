#!/usr/bin/env python3
""" that creates a bag of words embedding matrix: """
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


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
   # Initialize CountVectorizer with the given vocabulary
    vectorizer = CountVectorizer()

    if vocab is None:
        embeddings = vectorizer.fit_transform(sentences).toarray()
        vocab = vectorizer.get_feature_names_out()
    else:
        vectorizer = CountVectorizer(vocabulary=vocab)
        embeddings = vectorizer.fit_transform(sentences).toarray()
    
    return embeddings, vocab
