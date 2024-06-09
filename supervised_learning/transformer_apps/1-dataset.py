#!/usr/bin/env python3
""" loads and preps a dataset for machine translation:"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """ class Dataset """
    def __init__(self):
        """
            data_train, which contains the ted_hrlr_translate/pt_to_en
                tf.data.Dataset train split, loaded as_supervided
            data_valid, which contains the ted_hrlr_translate/pt_to_en
                tf.data.Dataset validate split, loaded as_supervided
            tokenizer_pt is the Portuguese tokenizer
                created from the training set
            tokenizer_en is the English tokenizer
                created from the training set
        """
        datasets = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'], as_supervised=True)
        self.data_train = datasets[0]
        self.data_valid = datasets[1]
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
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
            that encodes a translation into tokens:
            - pt is the tf.Tensor containing the Portuguese sentence
            - en is the tf.Tensor containing the corresponding English sentence
            - The tokenized sentences should include
                the start and end of sentence tokens
            The start token should be indexed as vocab_size
            The end token should be indexed as vocab_size + 1
                Returns: pt_tokens, en_tokens
                    pt_tokens is a np.ndarray containing the Portuguese tokens
                en_tokens is a np.ndarray. containing the English tokens

        """
        pt_size = self.tokenizer_pt.vocab_size
        en_size = self.tokenizer_en.vocab_size
        pt_tokens = self.tokenizer_pt.encode(pt.numpy())
        en_tokens = self.tokenizer_en.encode(en.numpy())
        pt_tokens = [pt_size] + pt_tokens + [pt_size + 1]
        en_tokens = [en_size] + en_tokens + [en_size + 1]
        return pt_tokens, en_tokens
