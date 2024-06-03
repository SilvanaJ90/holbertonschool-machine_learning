#!/usr/bin/env python3
"""
  that calculates the cumulative n-gram BLEU score for a sentence:
"""
import numpy as np
ngram_bleu = __import__('1-ngram_bleu').ngram_bleu


def cumulative_bleu(references, sentence, n):
    """

    - references is a list of reference translations
        each reference translation is a list of the words in the translation
    - sentence is a list containing the model proposed sentence
    - n is the size of the largest n-gram to use for evaluation
    - All n-gram scores should be weighted evenly
    Returns: the cumulative n-gram BLEU score

    """
    bleu_scores = []
    for i in range(1, n + 1):
        bleu_scores.append(ngram_bleu(references, sentence, i))
    return np.prod(bleu_scores) ** (1 / n) if bleu_scores else 0
