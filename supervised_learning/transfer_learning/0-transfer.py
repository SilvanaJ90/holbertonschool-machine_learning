#!/usr/bin/env python3
"""
 Write a python script that trains a convolutional
    neural network to classify the CIFAR 10 dataset:
"""
import tensorflow as tf
from tensorflow import keras as K
import numpy as np


def preprocess_data(X, Y):
    """
    Preprocesses the CIFAR 10 data.

    Args:
        X (numpy.ndarray): Input data of shape (m, 32, 32, 3),
        where m is the number of data points.
        Y (numpy.ndarray): Labels of shape (m,)
        corresponding to the input data X.

    Returns:
        X_p (numpy.ndarray): Preprocessed input data.
        Y_p (numpy.ndarray): Preprocessed labels.
    """
    X_p = K.applications.vgg16.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_valid, Y_valid = preprocess_data(X_valid, Y_valid)

    inputs = K.Input(shape=(32, 32, 3))
    inputs_resized = K.layers.Lambda(
        lambda x: K.backend.resize_images(x,
                                          height_factor=(224 / 32),
                                          width_factor=(224 / 32),
                                          data_format="channels_last"))(inputs)

    VGG16 = K.applications.VGG16(include_top=False,
                                 weights='imagenet',
                                 input_shape=(224, 224, 3))
    activation = K.activations.relu

    X = VGG16(inputs_resized)
    X = K.layers.GlobalAveragePooling2D()(X)
    X = K.layers.Dense(500, activation=activation)(X)
    X = K.layers.Dropout(0.2)(X)
    outputs = K.layers.Dense(10, activation='softmax')(X)

    model = K.Model(inputs=inputs, outputs=outputs)

    VGG16.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])

    history = model.fit(x=X_train, y=Y_train,
                        validation_data=(X_valid, Y_valid),
                        batch_size=300,
                        epochs=5, verbose=True)

    model.save('cifar10.keras')
