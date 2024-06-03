#!/usr/bin/env python3
"""
 that calculates the n-gram BLEU score for a sentence:
"""


def ngram_bleu(references, sentence, n):
    """
    - references is a list of reference translations
        each reference translation is a list of the words in the translation
    - sentence is a list containing the model proposed sentence
    - n is the size of the n-gram to use for evaluation
    Returns: the n-gram BLEU score

    """
