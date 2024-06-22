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
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Load the BERT model for question answering
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Tokenize the input question and reference text
    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    # Add the special tokens [CLS] and [SEP]
    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + reference_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(question_tokens) + 2) + [1] * (len(reference_tokens) + 1)

    # Pad the inputs if necessary (BERT requires fixed-length input)
    max_length = 512
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        input_mask = input_mask[:max_length]
        segment_ids = segment_ids[:max_length]
    else:
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length)

    # Convert inputs to tensors
    input_ids = tf.constant(input_ids)[None, :]
    input_mask = tf.constant(input_mask)[None, :]
    segment_ids = tf.constant(segment_ids)[None, :]

    # Run the model and get the start and end logits
    outputs = model([input_ids, input_mask, segment_ids])
    start_logits = outputs[0][0].numpy()
    end_logits = outputs[1][0].numpy()

    # Find the start and end positions of the answer
    start_position = tf.argmax(start_logits).numpy()
    end_position = tf.argmax(end_logits).numpy() + 1

    # Convert the tokens back to the original text
    answer_tokens = tokens[start_position:end_position]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if '[CLS]' in answer or '[SEP]' in answer or answer == '':
        return None
    return answer
