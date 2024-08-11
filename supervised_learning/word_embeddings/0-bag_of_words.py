#!/usr/bin/env python3
import numpy as np
import re
from collections import Counter

def tokenize(sentence):
    """
    Tokenize a sentence by removing punctuation and converting to lowercase.
    """
    sentence = re.sub(r"'s\b", '', sentence.lower())
    return re.findall(r'\b\w+\b', sentence)


def build_vocab(sentences, vocab=None):
    """
    Build the vocabulary list.
    If vocab is provided, use it as the vocabulary.
    Otherwise, create a vocabulary from sentences.
    """
    if vocab:
        return vocab
    
    # Create a counter for all words in sentences
    word_counter = Counter(word for sentence in sentences for word in tokenize(sentence))
    
    # Create vocabulary from counter keys
    return sorted(word_counter.keys())

def bag_of_words(sentences, vocab=None):
    """
    Create a bag of words embedding matrix.
    
    Arguments:
    sentences -- list of sentences to analyze
    vocab -- list of the vocabulary words to use for the analysis
    
    Returns:
    embeddings -- numpy.ndarray of shape (s, f) containing the embeddings
    features -- list of the features used for embeddings
    """
    # Build vocabulary
    features = build_vocab(sentences, vocab)
    vocab_size = len(features)
    
    # Create a dictionary to map words to their index in the feature list
    word_to_index = {word: index for index, word in enumerate(features)}
    
    # Initialize the embedding matrix
    embeddings = np.zeros((len(sentences), vocab_size), dtype=int)
    
    # Fill the embedding matrix
    for i, sentence in enumerate(sentences):
        word_counts = Counter(tokenize(sentence))
        for word, count in word_counts.items():
            if word in word_to_index:
                embeddings[i, word_to_index[word]] = count
    
    return embeddings, features