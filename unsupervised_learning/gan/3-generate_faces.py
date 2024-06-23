#!/usr/bin/env python3
""" Convolutional GenDiscr  """
import tensorflow as tf
from tensorflow.keras import layers, models


def convolutional_GenDiscr():
    """ Doc """

    def generator():
        """ DOc """
        model = models.Sequential()
        model.add(layers.Input(shape=(16,)))
        model.add(layers.Dense(2048, activation='tanh'))
        model.add(layers.Reshape((2, 2, 512)))

        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(64, kernel_size=3, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('tanh'))

        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(16, kernel_size=3, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('tanh'))

        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(1, kernel_size=3, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('tanh'))

        return model

    def discriminator():
        """ Doc """
        model = models.Sequential()
        model.add(layers.Input(shape=(16, 16, 1)))

        model.add(layers.Conv2D(32, kernel_size=3, padding='same'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Activation('tanh'))

        model.add(layers.Conv2D(64, kernel_size=3, padding='same'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Activation('tanh'))

        model.add(layers.Conv2D(128, kernel_size=3, padding='same'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Activation('tanh'))

        model.add(layers.Conv2D(256, kernel_size=3, padding='same'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Activation('tanh'))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    return generator(), discriminator()
