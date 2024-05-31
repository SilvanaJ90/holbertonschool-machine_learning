#!/usr/bin/env python3
""" creates an autoencoder: """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of
    the latent space representation
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization
    and binary cross-entropy loss
    All layers should use a relu activation except for the last layer
    in the decoder, which should use sigmoid
    """
    # Encoder
    encoder_input = keras.layers.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    encoder_output = keras.layers.Dense(latent_dims, activation='relu')(x)

    encoder = keras.models.Model(encoder_input, encoder_output, name='encoder')

    # Decoder
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.models.Model(decoder_input, decoder_output, name='decoder')

    # Autoencoder
    autoencoder_input = encoder_input
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    auto = keras.models.Model(autoencoder_input, decoded, name='autoencoder')

    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
