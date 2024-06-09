#!/usr/bin/env python3
""" Loads and preps a dataset for machine translation """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """ Class Dataset """
    def __init__(self, batch_size, max_len):
        """
            Initializes the dataset
            batch_size is the batch size for training/validation
            max_len is the maximum number of tokens
        """
        self.batch_size = batch_size
        self.max_len = max_len

        datasets = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'], as_supervised=True)
        self.data_train = datasets[0]
        self.data_valid = datasets[1]
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        self.data_train = self.data_train.filter(self.filter_max_length)
        self.data_valid = self.data_valid.filter(self.filter_max_length)

        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(10000)
        self.data_train = self.data_train.padded_batch(
            self.batch_size, padded_shapes=([None], [None]))
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.padded_batch(
            self.batch_size, padded_shapes=([None], [None]))

    def tokenize_dataset(self, data):
        """
            Tokenizes the dataset
            data is a tf.data.Dataset whose examples are formatted
                as a tuple (pt, en)
                pt is the tf.Tensor containing the Portuguese sentence
                en is the tf.Tensor containing English sentence
            The maximum vocab size should be set to 2**15
            Returns: tokenizer_pt, tokenizer_en
                tokenizer_pt is the Portuguese tokenizer
                tokenizer_en is the English tokenizer
        """
        pt_sentences = [pt.numpy() for pt, en in data]
        en_sentences = [en.numpy() for pt, en in data]

        pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            pt_sentences, target_vocab_size=2**15)
        en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            en_sentences, target_vocab_size=2**15)

        return pt, en

    def encode(self, pt, en):
        """
            Encodes a translation into tokens
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English sentence
            The tokenized sentences should include
                the start and end of sentence tokens
            The start token should be indexed as vocab_size
            The end token should be indexed as vocab_size + 1
            Returns: pt_tokens, en_tokens
                pt_tokens is a np.ndarray containing the Portuguese tokens
                en_tokens is a np.ndarray containing the English tokens
        """
        pt_size = self.tokenizer_pt.vocab_size
        en_size = self.tokenizer_en.vocab_size
        pt_tokens = self.tokenizer_pt.encode(pt.numpy())
        en_tokens = self.tokenizer_en.encode(en.numpy())
        pt_tokens = [pt_size] + pt_tokens + [pt_size + 1]
        en_tokens = [en_size] + en_tokens + [en_size + 1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
            Acts as a TensorFlow wrapper for the encode instance method
            Make sure to set the shape of the pt and en return tensors
        """
        pt_tokens, en_tokens = tf.py_function(
            self.encode,
            [pt, en], [tf.int64, tf.int64])
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens

    def filter_max_length(self, pt, en):
        """
            Filters out all examples that have either
            sentence with more than max_len tokens
        """
        return tf.logical_and(tf.size(pt) <= self.max_len,
                              tf.size(en) <= self.max_len)
