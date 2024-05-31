#!/usr/bin/env python3
"""  creates a variational autoencoder """
import tensorflow.keras as keras


import tensorflow as tf


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for
    each hidden layer in the encoder, respectively

    the hidden layers should be reversed for the decoder

    latent_dims is an integer containing the dimensions
    of the latent space representation
    Returns: encoder, decoder, auto

    encoder is the encoder model, which should output the latent
    representation,
    the mean, and the log variance, respectively
    decoder is the decoder model
    auto is the full autoencoder model

    The autoencoder model should be compiled using adam optimization
    and binary cross-entropy loss
    All layers should use a relu activation
    except for the mean and log variance
    layers in the encoder, which should use None, and the
    last layer in the decoder, which should use sigmoid
    """
    # Encoder
    inputs = keras.layers.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)
    z = keras.layers.Lambda(sampling, output_shape=(
        latent_dims,))([z_mean, z_log_var])
    encoder = keras.models.Model(
        inputs, [z, z_mean, z_log_var], name='encoder')

    # Decoder
    latent_inputs = keras.layers.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.models.Model(latent_inputs, outputs, name='decoder')

    # Autoencoder
    outputs = decoder(encoder(inputs)[0])
    auto = keras.models.Model(inputs, outputs, name='autoencoder')

    # Loss function
    reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    # Compile the model
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
