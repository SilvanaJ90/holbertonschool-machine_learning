#!/usr/bin/env python3
"""  that creates a convolutional autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """

    input_dims is a tuple of integers containing the
    dimensions of the model input
    filters is a list containing the number of filters for each
    convolutional layer in the encoder, respectively
        the filters should be reversed for the decoder
    latent_dims is a tuple of integers containing the
    dimensions of the latent space representation
    Each convolution in the encoder should use a kernel size
    of (3, 3) with same padding and relu activation,
    followed by max pooling of size (2, 2)
    Each convolution in the decoder, except for the last two, should use a
    filter size of (3, 3) with same padding and relu activation,
    followed by upsampling of size (2, 2)
        The second to last convolution should instead use valid padding
        The last convolution should have the same number of
        filters as the number of channels in input_dims with
        sigmoid activation and no upsampling
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization
    and binary cross-entropy loss

    """
    encoded_input = keras.layers.Input(shape=input_dims)
    x = encoded_input
    for num_filters in filters:
        x = keras.layers.Conv2D(
            num_filters, (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    # Latent space
    encoded_output = x

    # Decoder
    decoded_input = keras.layers.Input(shape=latent_dims)
    x = decoded_input

    for num_filters in reversed(filters[:-1]):
        x = keras.layers.Conv2D(
            num_filters, (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(
        filters[-1], (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    decoded_output = keras.layers.Conv2D(
        1, (3, 3), activation='sigmoid', padding='same')(x)

    encoder = keras.models.Model(
        inputs=encoded_input, outputs=encoded_output)
    decoder = keras.models.Model(
        inputs=decoded_input, outputs=decoded_output)
    # Autoencoder
    autoencoder_output = decoder(encoder(encoded_input))
    autoencoder = keras.models.Model(
        inputs=encoded_input, outputs=autoencoder_output)

    # Compile the autoencoder model
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
