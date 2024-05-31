#!/usr/bin/env python3
""" Script to forecast BTC prices using an LSTM model """
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
preprocess_data = __import__('preprocess_data').preprocess_data


def create_dataset(df, sequence_length, batch_size):
    """ Convert tf.data.Dataset """
    data = df.values
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=len(x)).batch(batch_size)
    return dataset


def plot_predictions(test_df, predictions, train_mean, train_std):
    """ Function to Plot Predictions"""
    predictions = (predictions * train_std['Close']) + train_mean['Close']
    test_df['Close'] = \
        (test_df['Close'] * train_std['Close']) + train_mean['Close']

    plt.figure(figsize=(10, 6))
    plt.plot(test_df.index[sequence_length:],
             test_df['Close'][sequence_length:],
             color='b', label='Actual Data')
    plt.plot(test_df.index[sequence_length:],
             predictions, color='r',
             label='Predictions', zorder=5)
    plt.xlabel('Date')
    plt.ylabel('BTC Price')
    plt.title('BTC Price Forecasting')
    plt.legend()
    plt.show()


# Load preprocessed data
data_folder = 'data'
coinbase_path = os.path.join(data_folder, 'coinbase.csv')
bitstamp_path = os.path.join(data_folder, 'bitstamp.csv')
train_df, val_df, test_df, train_mean, train_std = \
    preprocess_data(coinbase_path, bitstamp_path)

sequence_length = 24  # Number of hours to look back for prediction
batch_size = 64

train_dataset = create_dataset(train_df, sequence_length, batch_size)
val_dataset = create_dataset(val_df, sequence_length, batch_size)
test_dataset = create_dataset(test_df, sequence_length, batch_size)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(
        sequence_length, train_df.shape[1]), return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(train_dataset, validation_data=val_dataset, epochs=10)

mse = model.evaluate(test_dataset)
print(f"Test MSE: {mse}")

# Make predictions
test_data = test_df.values
x_test = []
for i in range(len(test_data) - sequence_length):
    x_test.append(test_data[i:i + sequence_length])
x_test = np.array(x_test)
predictions = model.predict(x_test)

# Plot predictions
plot_predictions(test_df, predictions, train_mean, train_std)
