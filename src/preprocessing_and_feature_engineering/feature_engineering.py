import pandas as pd
import numpy as np


def create_sequences(data, lookback=12):
    xs, ys = [], []
    for i in range(len(data) - lookback):
        xs.append(data[i : i + lookback])
        ys.append(data[i + lookback])
    return np.array(xs), np.array(ys)