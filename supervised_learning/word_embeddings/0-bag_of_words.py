#!/usr/bin/env python3
""" that creates a bag of words embedding matrix: """
import numpy as np
import re
from collections import Counter


def preprocess(sentence):
    """ Lowercase and remove punctuation """
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence

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
    processed_sentences = [preprocess(sentence).split() for sentence in sentences]
    
    if vocab is None:
        # Create a vocabulary from all unique words in the sentences
        all_words = [word for sentence in processed_sentences for word in sentence]
        vocab = list(set(all_words))
    
    # Create a vocabulary index mapping
    vocab_index = {word: idx for idx, word in enumerate(sorted(vocab))}
    
    # Initialize embeddings matrix
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    
    # Fill in the embeddings matrix
    for i, sentence in enumerate(processed_sentences):
        word_count = Counter(sentence)
        for word, count in word_count.items():
            if word in vocab_index:
                embeddings[i, vocab_index[word]] = count
    
    return embeddings, sorted(vocab)