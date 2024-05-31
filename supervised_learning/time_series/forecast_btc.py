#!/usr/bin/env python3
""" Script forescast """
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def create_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    # Load preprocessed data
    X = np.load('X.npy')
    y = np.load('y.npy')

    # Split the data into training and validation sets
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Create tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64).prefetch(1)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(64).prefetch(1)

    # Create and train the model
    model = create_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[early_stopping])

    # Save the trained model
    model.save('btc_forecast_model.h5')

if __name__ == "__main__":
    main()