#!/usr/bin/env python3
""" answers questions from multiple reference texts: """
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(coprus_path):
    """
    corpus_path is the path to the corpus of reference documents
    """
    w = ['exit', 'quit', 'goodbye', 'bye']
    while True:
        question = input('Q: ')
        if question.lower() in w:
            print('A: Goodbye')
            break
        else:
            reference = semantic_search(coprus_path)
            answer = question_answer(question, reference)
            if answer is None or answer == '':
                print('A: Sorry, I do not understand your question.')
            else:
                print('A: {}'.format(answer))
