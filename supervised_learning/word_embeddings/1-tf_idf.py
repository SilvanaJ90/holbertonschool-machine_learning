#!/usr/bin/env python3
"""  that creates a TF-IDF embedding: """
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    - sentences is a list of sentences to analyze
    - vocab is a list of the vocabulary words to use for the analysis
        If None, all words within sentences should be used
    Returns: embeddings, features
        embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
            s is the number of sentences in sentences
            f is the number of features analyzed
        features is a list of the features used for embeddings
    """
    tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab)

    ouptup = tfidf_vectorizer.fit_transform(sentences)
    return ouptup.toarray(), tfidf_vectorizer.get_feature_names()
