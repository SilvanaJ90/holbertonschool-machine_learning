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
    # Calculate precision
    word_max_count = {}
    for word in set(sentence):
        for reference in references:
            reference_word_count = reference.count(word)
            if reference_word_count > word_max_count.get(word, 0):
                word_max_count[word] = reference_word_count
    precision = sum(word_max_count.values())

    # Calculate brevity penalty
    closest_ref_len = min(len(reference) for reference in references)
    sentence_len = len(sentence)
    brevity_penalty = 1 if sentence_len >= closest_ref_len \
        else np.exp(1 - (closest_ref_len / sentence_len))

    # Calculate BLEU score
    bleu_score = precision * brevity_penalty / sentence_len

    return bleu_score
