#!/usr/bin/env python3
import pandas as pd


data = {
    'first': [0.0, 0.5, 1.0, 1.5],
    'seconds': ['one', 'two', 'three', 'four']
}
index = ['A', 'B', 'C', 'D']
df = pd.DataFrame(data, index=index)
