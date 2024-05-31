#!/usr/bin/env python3
""" rains a model using mini-batch gradient descent """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
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

    if learning_rate_decay and validation_data:
        def lr_schedule(epoch):
            """Función de programación de la tasa de aprendizaje"""
            return alpha / (1 + decay_rate * epoch)

        lr_scheduler_callback = K.callbacks.LearningRateScheduler(
            lr_schedule, verbose=1)
        callbacks.append(lr_scheduler_callback)

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
