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
    precision = 0
    sentence_len = len(sentence)
    closest_ref_len = min(len(reference) for reference in references)

    for word in sentence:
        max_word_count = max(reference.count(word) for reference in references)
        precision += max_word_count

    precision /= len(sentence)

    brevity_penalty = 1
    if sentence_len < closest_ref_len:
        brevity_penalty = np.exp(1 - (closest_ref_len / sentence_len))

    bleu_score = brevity_penalty * precision
    return bleu_score
