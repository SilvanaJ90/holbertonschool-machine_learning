#!/usr/bin/env python3
"""that performs semantic search on a corpus of documents: """
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """

    - corpus_path is the path to the corpus of reference
        documents on which to perform semantic search
    - sentence is the sentence from which to perform semantic search
    Returns: the reference text of the document most similar to sentence
    """
    reference = []
    reference.append(sentence)
    dirs = os.listdir(corpus_path)
    for file in dirs:
        if not file.endswith('.md'):
            continue
        with open(corpus_path+'/'+file, 'r', encoding='utf-8') as f:
            reference.append(f.read())
    e = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    embeddings = e(reference)
    corr = np.inner(embeddings, embeddings)
    close = np.argmax(corr[0, 1:])
    return reference[close + 1]
