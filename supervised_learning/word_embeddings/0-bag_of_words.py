#!/usr/bin/env python3
import numpy as np
import re

def bag_of_words(sentences, vocab=None):
    # Limpiar y tokenizar las oraciones
    def tokenize(sentence):
        sentence = re.sub(r"\b's\b", '', sentence.lower())  # Eliminar contracciones
        return re.findall(r'\b\w+\b', sentence)  # Tokenizar
    
    # Generar vocabulario si no está proporcionado
    if vocab is None:
        vocab_set = set()
        for sentence in sentences:
            words = tokenize(sentence)
            vocab_set.update(words)
        vocab = sorted(list(vocab_set))
    
    # Crear la matriz de embeddings
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    
    for i, sentence in enumerate(sentences):
        words = tokenize(sentence)
        for word in words:
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1
    
    return embeddings, vocab