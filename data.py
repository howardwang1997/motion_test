# coding=utf-8

import pandas as pd
import os

# os.listdir(../../)
path = '../../data/snowboard/20250524/20250524125735-后刃落叶飘.txt'
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


def read_data(path):
    df = pd.read_csv(path, sep='\t')
    df.iloc[:, 0] = df.iloc[:, 0].apply(convert_time)

    # # Verify results
    # print("DataFrame successfully created!")
    # print(f"Shape: {df.shape}")
    # # print(type(df.iloc[0,0]))
    
    return df

