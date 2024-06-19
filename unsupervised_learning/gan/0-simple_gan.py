#!/usr/bin/env python3
""" Simple GAN """
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """ Class Simple GAN """
    def __init__(self, generator, discriminator,
                 latent_generator, real_examples,
                 batch_size=200, disc_iter=2, learning_rate=.005):
        """ INIT """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: \
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer,
                               loss=generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: \
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) + \
            tf.keras.losses.MeanSquaredError()(y, -1*tf.ones(y.shape))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss)

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # overloading train_step()
    def train_step(self, useless_argument):
        for _ in range(self.disc_iter):
            # Compute the loss for the discriminator in a
            # tape watching the discriminator's weights
            with tf.GradientTape() as disc_tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample()
                disc_real_output = self.discriminator(
                    real_sample, training=True)
                disc_fake_output = self.discriminator(
                    fake_sample, training=True)
                discr_loss = self.discriminator.loss(
                    disc_real_output, disc_fake_output)

            # Apply gradient descent once to the discriminator
            disc_gradients = disc_tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(disc_gradients, self.discriminator.trainable_variables))

        # Compute the loss for the generator in
        # a tape watching the generator's weights
        with tf.GradientTape() as gen_tape:
            fake_sample = self.get_fake_sample()
            gen_output = self.discriminator(fake_sample, training=False)
            gen_loss = self.generator.loss(gen_output)

        # Apply gradient descent to the generator
        gen_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
