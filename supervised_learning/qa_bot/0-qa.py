#!/usr/bin/env python3
"""
    that finds a snippet of text within a
    reference document to answer a question:
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')

    # Load the BERT model for question answering
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Tokenize the input question and reference text
    reference_tokens = tokenizer.tokenize(reference)
    reference_tokens += ['[SEP]']
    reference_tokens_ids = tokenizer.convert_tokens_to_ids(reference_tokens)

    question_tokens = tokenizer.tokenize(question)
    question_tokens = ['[CLS]'] + question_tokens + ['[SEP]']
    question_tokens_ids = tokenizer.convert_tokens_to_ids(question_tokens)
    question_tokens_ids = tokenizer.convert_tokens_to_ids(question_tokens)
    input_ids = question_tokens_ids + reference_tokens_ids
    input_mask = [1] * len(input_ids)
    input_types = [0] * len(question_tokens) + [1] * len(reference_tokens)
    input_ids, input_mask, input_types = map(
        lambda t: tf.expand_dims(
            tf.convert_to_tensor(
                t, dtype=tf.int32), 0), (input_ids, input_mask, input_types))
    outputs = model([input_ids, input_mask, input_types])
    short_start = tf.argmax(outputs[0][0][1:-1]) + 1
    short_end = tf.argmax(outputs[1][0][1:-1]) + 1
    tokens = question_tokens + reference_tokens
    answer_tokens = tokens[short_start: short_end + 1]
    if len(answer_tokens) is 0:
        answer = None
    else:
        answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer
