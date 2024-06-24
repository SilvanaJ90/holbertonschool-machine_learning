#!/usr/bin/env python3
""" Convolutional GenDiscr  """
import tensorflow as tf


class WGAN_GP:
    def __init__(self, generator, discriminator, latent_generator, real_examples, batch_size=64, disc_iter=5, learning_rate=0.0001):
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.history = None

    def compile(self):
        # Define optimizers
        self.gen_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.0, beta_2=0.9)
        self.disc_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.0, beta_2=0.9)
        
        # Define loss functions
        self.gen_loss_fn = self.generator_loss
        self.disc_loss_fn = self.discriminator_loss

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = -tf.reduce_mean(real_output)
        fake_loss = tf.reduce_mean(fake_output)
        return real_loss + fake_loss

    def replace_weights(self, gen_h5, disc_h5):
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)

    def get_fake_sample(self, n):
        latent_samples = self.latent_generator(n)
        return self.generator(latent_samples)

    def get_real_sample(self, n):
        idx = np.random.randint(0, self.real_examples.shape[0], n)
        return self.real_examples[idx]
