#!/usr/bin/env python3
"""
Convolutional GenDiscr
This script contains the WGAN_GP class definition,
which implements a Wasserstein GAN with Gradient Penalty.
"""
import tensorflow as tf
import numpy as np


class WGAN_GP:
    """
    Class implementing a Wasserstein GAN with Gradient Penalty.

    Attributes:
    generator (tf.keras.Model): The generator model.
    discriminator (tf.keras.Model): The discriminator model.
    latent_generator (function): Function to generate latent samples.
    real_examples (tf.Tensor): Tensor containing real examples for training.
    batch_size (int): The size of the batch for training. Default is 64.
    disc_iter (int): Number of iterations to update
    the discriminator per generator update. Default is 5.
    learning_rate (float): Learning rate for the optimizers. Default is 0.0001.
    history (None): Placeholder for storing training history.
    """
    def __init__(self, generator, discriminator,
                 latent_generator, real_examples,
                 batch_size=64, disc_iter=5, learning_rate=0.0001):
        """
        Initializes the WGAN_GP class with the provided models and parameters.

        Args:
        generator (tf.keras.Model): The generator model.
        discriminator (tf.keras.Model): The discriminator model.
        latent_generator (function): Function to generate latent samples.
        real_examples (tf.Tensor): Tensor containing
        real examples for training.
        batch_size (int, optional): The size of the batch
        for training. Default is 64.
        disc_iter (int, optional): Number of iterations to
        update the discriminator per generator update. Default is 5.
        learning_rate (float, optional): Learning rate for the
        optimizers. Default is 0.0001.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.history = None

    def compile(self):
        """
        Compiles the WGAN_GP model by defining
        the optimizers and loss functions.
        """
        # Define optimizers
        self.gen_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate, beta_1=0.0, beta_2=0.9)
        self.disc_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate, beta_1=0.0, beta_2=0.9)

        # Define loss functions
        self.gen_loss_fn = self.generator_loss
        self.disc_loss_fn = self.discriminator_loss

    def generator_loss(self, fake_output):
        """
        Computes the loss for the generator.

        Args:
        fake_output (tf.Tensor): Output from the
        discriminator for generated samples.

        Returns:
        tf.Tensor: The generator loss.
        """
        return -tf.reduce_mean(fake_output)

    def discriminator_loss(self, real_output, fake_output):
        """
        Computes the loss for the discriminator.

        Args:
        real_output (tf.Tensor): Output from the
        discriminator for real samples.
        fake_output (tf.Tensor): Output from the
        discriminator for generated samples.

        Returns:
        tf.Tensor: The discriminator loss.
        """
        real_loss = -tf.reduce_mean(real_output)
        fake_loss = tf.reduce_mean(fake_output)
        return real_loss + fake_loss

    def replace_weights(self, gen_h5, disc_h5):
        """
        Replaces the weights of the generator
        and discriminator with the provided weights.

        Args:
        gen_h5 (str): Path to the .h5 file
        containing the generator weights.
        disc_h5 (str): Path to the .h5 file
        containing the discriminator weights.
        """
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)

    def get_fake_sample(self, n):
        """
        Generates fake samples using the generator.

        Args:
        n (int): Number of fake samples to generate.

        Returns:
        tf.Tensor: Tensor containing the generated fake samples.
        """
        latent_samples = self.latent_generator(n)
        return self.generator(latent_samples)

    def get_real_sample(self, n):
        """
        Retrieves a batch of real samples from the dataset.

        Args:
        n (int): Number of real samples to retrieve.

        Returns:
        tf.Tensor: Tensor containing the real samples.
        """
        idx = np.random.randint(0, self.real_examples.shape[0], n)
        return self.real_examples[idx]
