#!/usr/bin/env python3
"""
 that calculates the n-gram BLEU score for a sentence:
"""
import numpy as np
from collections import Counter


def count_ngrams(tokens, n):
    """Count n-grams in a list of tokens."""
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def modified_ngram_precision(references, candidate, n):
    """Calculate modified n-gram precision."""
    candidate_counts = count_ngrams(candidate, n)
    max_counts = {}
    for reference in references:
        reference_counts = count_ngrams(reference, n)
        for ngram in candidate_counts:
            max_counts[ngram] = max(
                max_counts.get(ngram, 0), reference_counts[ngram])
    total_ngram_count = sum(candidate_counts.values())
    clipped_count = sum(min(
        candidate_counts[ngram],
        max_counts.get(ngram, 0)) for ngram in candidate_counts)
    return clipped_count / total_ngram_count if total_ngram_count > 0 else 0


def brevity_penalty(references, candidate):
    """Calculate brevity penalty."""
    reference_lengths = [len(reference) for reference in references]
    closest_ref_len = min(
        reference_lengths, key=lambda x: abs(len(candidate) - x))
    if len(candidate) > closest_ref_len:
        return 1
    else:
        return np.exp(1 - closest_ref_len / len(candidate))


def ngram_bleu(references, sentence, n):
    """
    Calculate the n-gram BLEU score for a sentence.

    Args:
        references (list): A list of reference translations.
            Each reference translation is
            a list of the words in the translation.
        sentence (list): A list containing the model proposed sentence.
        n (int): The size of the n-gram to use for evaluation.

    Returns:
        float: The n-gram BLEU score.
    """
    precision = modified_ngram_precision(references, sentence, n)
    bp = brevity_penalty(references, sentence)
    bleu = bp * precision
    return bleu
