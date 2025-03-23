from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.pytorchforecasting import PytorchForecastingNHiTS
from sktime.utils._testing.hierarchical import _make_hierarchical
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, header=0, index_col=0)
    df.columns = df.columns.str.strip()
    df = df[df.columns[::-1]]  # reverse columns so dates ascend
    row_label = "Standard Average Hotel Occupancy Rate (Per Cent)"
    if row_label not in df.index:
        raise ValueError(f"Row '{row_label}' not found.")
    total_series = df.loc[row_label]
    print(total_series)
    total_df = total_series.reset_index()
    total_df.columns = ['Date', 'HotelOccupancy']
    total_df['Date'] = pd.to_datetime(total_df['Date'], format='%Y %b')
    total_df = total_df.sort_values('Date')
    total_df['HotelOccupancy'] = pd.to_numeric(total_df['HotelOccupancy'], errors='coerce')
    total_df.dropna(subset=['HotelOccupancy'], inplace=True)
    print(total_df)
    return total_df


def split_data(total_df):
    train = total_df[(total_df['Date'] >= '2013-01-01') & (total_df['Date'] <= '2018-12-31')]
    test = total_df[(total_df['Date'] >= '2019-01-01') & (total_df['Date'] <= '2019-12-31')]
    return train, test


def extract_datetime_features(df):
    df = df.copy()
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['year_offset'] = df['year'] - df['year'].min()
    return df[['month_sin', 'month_cos', 'year_offset']]


def main():
    filepath = '../Datasets/hotel_bookings.csv'
    total_df = load_and_preprocess_data(filepath)
    train_df, test_df = split_data(total_df)
    y_train = train_df.set_index("Date")["HotelOccupancy"]
    y_test = test_df.set_index("Date")["HotelOccupancy"]
    X_train = extract_datetime_features(train_df).set_index(train_df["Date"])
    X_test = extract_datetime_features(test_df).set_index(test_df["Date"])
    fh = ForecastingHorizon(range(1, len(y_test) + 1), is_relative=True)
    model = PytorchForecastingNHiTS(trainer_params={"max_epochs": 40})
    model.fit(y=y_train, X=X_train, fh=fh)
    y_pred = model.predict(fh=fh, X=X_test)
    plt.figure(figsize=(20, 12))
    plt.plot(y_test.index, y_test, marker="o", label="Actual")
    plt.plot(y_test.index, y_pred, marker="o", label="Forecast")
    plt.xlabel("Date")
    plt.ylabel("Hotel Occupancy")
    plt.title("Hotel Occupancy Forecast vs Actual")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
