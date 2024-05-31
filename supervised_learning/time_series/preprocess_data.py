#!/usr/bin/env python3
""" Script to preprocess data """
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os


data_folder = 'data'
coinbase_path = os.path.join(data_folder, 'coinbase.csv')
bitstamp_path = os.path.join(data_folder, 'bitstamp.csv')


def preprocess_data(coinbase_path, bitstamp_path):
    """ Load the datasets """
    df1 = pd.read_csv(coinbase_path)
    df2 = pd.read_csv(bitstamp_path)

    # Convert 'Timestamp' to datetime
    df1['Timestamp'] = pd.to_datetime(df1['Timestamp'], unit='s')
    df2['Timestamp'] = pd.to_datetime(df2['Timestamp'], unit='s')

    # Merge the two datasets on 'Timestamp'
    df = pd.merge(df1, df2, on='Timestamp',
                  suffixes=('_bitstamp', '_coinbase'))

    # Replace NaN values in df1 with values from df2
    df['Close'] = df['Close_bitstamp'].combine_first(df['Close_coinbase'])
    df['Weighted_Price'] = df['Weighted_Price_bitstamp'].combine_first(
        df['Weighted_Price_coinbase'])

    # Keep only relevant columns
    df = df[['Timestamp', 'Close', 'Weighted_Price']]

    # Filter for 'Timestamp' >= '2017'
    df = df[df['Timestamp'] >= "2017"]

    # Set 'Timestamp' as the index and resample to hourly data
    df.set_index('Timestamp', inplace=True)
    df = df.resample('h').mean()

    print(df.head())

    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]
    num_features = df.shape[1]

    train_mean = train_df.mean()
    train_std = train_df.std()
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    train_df = train_df.diff().dropna()
    val_df = val_df.diff().dropna()
    test_df = test_df.diff().dropna()
    print(train_df.shape)
    print(val_df.shape)
    print(test_df.shape)
    return train_df, val_df, test_df, train_mean, train_std


if __name__ == "__main__":
    preprocess_data(coinbase_path, bitstamp_path)
