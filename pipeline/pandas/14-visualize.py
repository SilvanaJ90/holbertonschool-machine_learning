#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Complete the following script to visualize the pd.DataFrame:
# The column Weighted_Price should be removed
df = df.drop(columns=['Weighted_Price'])

# Rename the column Timestamp to Date
df = df.rename(columns={'Timestamp' : 'Date'})

# Convert the timestamp values to date values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index the data frame on Date
df = df.set_index('Date')

# Missing values in Close should be set to the previous row value
df['Close'] = df['Close'].fillna(method='pad')

# Missing values in High, Low, Open should be set to the same row’s Close value
# Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df = df.fillna({
    'High': df['Close'],
    'Low': df['Close'],
    'Open': df['Close'],
    'Volume_(BTC)': 0,
    'Volume_(Currency)': 0
})


# Plot the data from 2017 and beyond at daily intervals and group the values of the same day such that:
    # High: max
    # Low: min
    # Open: mean
    # Close: mean
    # Volume(BTC): sum
    # Volume(Currency): sum

df.loc[pd.to_datetime('1-1-2017'):] \
    .resample('1440 min')\
    .aggregate({
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    })\
    .plot()
plt.show(block=True)


