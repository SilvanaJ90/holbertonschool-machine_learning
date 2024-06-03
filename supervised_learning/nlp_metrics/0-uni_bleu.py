#!/usr/bin/env python3
"""  that calculates the unigram BLEU score for a sentence: """
import numpy as np


def uni_bleu(references, sentence):
    """
    - references is a list of reference translations
        each reference translation is a list of the words in the translation
    - sentence is a list containing the model proposed sentence
    Returns: the unigram BLEU score

    """
    word_max_count = {}
    for word in set(sentence):
        for reference in references:
            reference_word_count = reference.count(word)
            if reference_word_count > word_max_count.get(word, 0):
                word_max_count[word] = reference_word_count
    min_reference_len = min(len(reference) for ref in reference)
    sentence_len = len(sentence)
    brevity_penalty = 1 if (sentence_len > min_reference_len) \
        else np.exp(1 - (min_reference_len / sentence_len))

    return np.exp(1 - (min_reference_len / sentence_len))
