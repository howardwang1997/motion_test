# coding=utf-8

import numpy as np
import pandas as pd
import os

# os.listdir(../../)
path = '../../data/snowboard/20250524_split/20250524132532-S型_0.csv'
# Read the file into a DataFrame
# df = pd.read_csv(path, sep='\t')

# Convert the timestamp column to datetime
def convert_time(time_str):
    try:
        # Split into date and time components
        date_part, time_part = time_str.split(' ', 1)
        hour, minute, second, millisecond = time_part.split(':')
        # Pad single-digit seconds to two digits
        second = second.zfill(2)
        # Reconstruct into ISO format
        new_time_str = f"{date_part} {hour}:{minute}:{second}.{millisecond}"
        return pd.to_datetime(new_time_str, format='%Y-%m-%d %H:%M:%S.%f')
    except (ValueError, AttributeError):
        return pd.NaT


def read_data(path: str): #-> pd.DataFrame
    if 'txt' in path:
        df = pd.read_csv(path, sep='\t')
    else:
        df = pd.read_csv(path)
    df.iloc[:, 0] = df.iloc[:, 0].apply(convert_time)

    # # Verify results
    # print("DataFrame successfully created!")
    # print(f"Shape: {df.shape}")
    # # print(type(df.iloc[0,0]))
    
    return df


def make_dataset(data: pd.DataFrame, window=100): #-> np.array
    data = np.array(data)
    n_points = data.shape[0] // window
    dataset = np.empty((1, 9*window))
    for i in range(n_points):
        try:
            dataset = np.vstack([dataset, data[int(i*window/2): int((i+2)*window/2), [2,3,4,5,6,7,11,12,13]].reshape(1, -1)])
        except ValueError:
            print(dataset.shape, data[int(i*window): int((i+1)*window), [2,3,4,5,6,7,11,12,13]].reshape(1, -1).shape)
            # print(data[int(i*window/2): int((i+2)*window/2), [2,3,4,5,6,7,11,12,13]].reshape(1, -1))

    # print('shape of dataset:', dataset.shape)
    return dataset[1:]
