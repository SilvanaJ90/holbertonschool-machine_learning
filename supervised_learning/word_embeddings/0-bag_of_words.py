#!/usr/bin/env python3
from nltk.stem import WordNetLemmatizer
import string

def preprocess(sentence):
    # Convertir a minúsculas y eliminar puntuación
    sentence = sentence.lower()
    sentence = ''.join(char for char in sentence if char not in string.punctuation)
    return sentence

def singularize(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word)

def bag_of_words(sentences):
    # Preprocesar y singularizar
    processed_sentences = [preprocess(sentence).split() for sentence in sentences]
    all_words = [singularize(word) for sentence in processed_sentences for word in sentence]
    
    # Crear vocabulario único
    vocab = list(set(all_words))
    vocab.sort()
    
    # Crear matriz de representaciones
    E = []
    for sentence in processed_sentences:
        row = [1 if word in sentence else 0 for word in vocab]
        E.append(row)
    
    return E, vocab