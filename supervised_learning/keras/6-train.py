#!/usr/bin/env python3
""" rains a model using mini-batch gradient descent """
import tensorflow.keras as K


def train_model(
        network, data, labels,
        batch_size, epochs,
        validation_data=None,
        early_stopping=False,
        patience=0,
        verbose=True,
        shuffle=False):
    """
    Doc

    """

    callbacks = []
    if early_stopping:
        early_stopping_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stopping_callback)

        return network.fit(
            x=data,
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=shuffle
            )
