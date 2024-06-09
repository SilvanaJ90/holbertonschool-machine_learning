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
